
import calendar
import csv
import json
import os
import re
import base64
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_experimental.text_splitter import SemanticChunker


# LlamaIndex core components
from llama_index.core import (
    VectorStoreIndex,    # Standard embedding-based index
    SummaryIndex,        # Full document storage, LLM-based relevance
    TreeIndex,           # Hierarchical summarization tree
    SimpleKeywordTableIndex,
    Document as LlamaDocument,
    Settings             # Global configuration
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Ensure LangSmith tracing is enabled
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
# def query_HRAgent():

# def query_ITAgent():

# def create_jiraticket(email:str,summary:str, description:str,category:str,priority:str):

# def check_ticketstatus(query:str):


class JiraIntegrationError(RuntimeError):
    """Raised when Jira configuration or issue creation fails."""


def create_jiraticket(
    email: str,
    summary: str,
    description: str,
    category: str,
    priority: str,
) -> dict:
    """Create an issue in the configured Jira Cloud project."""
    fields = {
        "email": email.strip(),
        "summary": summary.strip(),
        "description": description.strip(),
        "category": category.strip(),
        "priority": priority.strip(),
    }
    missing = [name for name, value in fields.items() if not value]
    if missing:
        raise ValueError(f"Please provide: {', '.join(missing)}.")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", fields["email"]):
        raise ValueError("Enter a valid email address.")

    base_url = os.getenv("JIRA_BASE_URL", "").strip().rstrip("/")
    jira_email = os.getenv("JIRA_EMAIL", "").strip()
    api_token = os.getenv("JIRA_API_TOKEN", "").strip()
    project_key = os.getenv("JIRA_PROJECT_KEY", "").strip()
    issue_type = os.getenv("JIRA_ISSUE_TYPE", "Task").strip()
    missing_config = [
        name for name, value in (
            ("JIRA_BASE_URL", base_url),
            ("JIRA_EMAIL", jira_email),
            ("JIRA_API_TOKEN", api_token),
            ("JIRA_PROJECT_KEY", project_key),
            ("JIRA_ISSUE_TYPE", issue_type),
        )
        if not value
    ]
    if missing_config:
        raise JiraIntegrationError(
            "Configure the following environment variables before submitting "
            f"to Jira: {', '.join(missing_config)}."
        )

    parsed_url = urlsplit(base_url)
    if (
        parsed_url.scheme != "https"
        or not parsed_url.netloc
        or parsed_url.username
        or parsed_url.password
        or parsed_url.query
        or parsed_url.fragment
        or parsed_url.path not in ("", "/")
    ):
        raise JiraIntegrationError(
            "JIRA_BASE_URL must be an HTTPS Jira site URL without credentials, "
            "a path, query, or fragment."
        )

    default_priority_map = {
        "Urgent": "Highest",
        "High": "High",
        "Medium": "Medium",
        "Low": "Low",
    }
    priority_map = default_priority_map
    configured_priority_map = os.getenv("JIRA_PRIORITY_MAP")
    if configured_priority_map:
        try:
            priority_map = json.loads(configured_priority_map)
        except json.JSONDecodeError as exc:
            raise JiraIntegrationError(
                "JIRA_PRIORITY_MAP must be a JSON object mapping support "
                "priorities to Jira priority names."
            ) from exc
        if not isinstance(priority_map, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in priority_map.items()
        ):
            raise JiraIntegrationError(
                "JIRA_PRIORITY_MAP must map string priority names to string "
                "Jira priority names."
            )

    jira_priority = priority_map.get(fields["priority"])
    if not jira_priority:
        raise JiraIntegrationError(
            f"No Jira priority is configured for '{fields['priority']}'. "
            "Set JIRA_PRIORITY_MAP in the environment."
        )

    category_label = re.sub(
        r"[^a-z0-9-]+",
        "-",
        fields["category"].lower(),
    ).strip("-")
    description_text = (
        f"Requester email: {fields['email']}\n"
        f"Category: {fields['category']}\n"
        f"Support priority: {fields['priority']}\n\n"
        f"{fields['description']}"
    )
    issue_fields = {
        "project": {"key": project_key},
        "summary": fields["summary"],
        "description": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": description_text}],
                }
            ],
        },
        "issuetype": {"name": issue_type},
        "priority": {"name": jira_priority},
        "labels": ["smarthr", category_label] if category_label else ["smarthr"],
    }
    request_body = json.dumps({"fields": issue_fields}).encode("utf-8")
    credentials = base64.b64encode(
        f"{jira_email}:{api_token}".encode("utf-8")
    ).decode("ascii")
    request = Request(
        f"{base_url}/rest/api/3/issue",
        data=request_body,
        headers={
            "Authorization": f"Basic {credentials}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=20) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        try:
            details = json.loads(error_body)
            error_message = "; ".join(
                details.get("errorMessages", [])
                + list(details.get("errors", {}).values())
            )
        except (json.JSONDecodeError, AttributeError):
            error_message = error_body[:1000]
        message = error_message or f"Jira returned HTTP {exc.code}."
        raise JiraIntegrationError(
            f"Jira did not create the issue: {message}"
        ) from exc
    except URLError as exc:
        raise JiraIntegrationError(
            f"Could not connect to Jira: {exc.reason}"
        ) from exc
    except (TimeoutError, OSError) as exc:
        raise JiraIntegrationError(
            f"Could not complete the Jira request: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise JiraIntegrationError(
            "Jira returned an invalid response after issue creation."
        ) from exc

    issue_key = response_data.get("key")
    if not issue_key:
        raise JiraIntegrationError(
            "Jira returned success without an issue key; check the Jira "
            "project to confirm whether the issue was created."
        )
    return {
        "status": "created",
        "submitted": True,
        "issue": {
            "key": issue_key,
            "url": f"{base_url}/browse/{issue_key}",
            "project_key": project_key,
            "summary": fields["summary"],
        },
    }


def create_jiraticket_tool(tool_input: str) -> str:
    """Adapt the ticket-draft function to the agent's single-string tool API."""
    try:
        fields = json.loads(tool_input)
    except json.JSONDecodeError:
        return (
            "Ticket draft not created. Provide the required fields as a JSON "
            "object: email, summary, description, category, priority."
        )

    if not isinstance(fields, dict):
        return "Ticket draft not created. The tool input must be a JSON object."

    required_fields = ("email", "summary", "description", "category", "priority")
    missing = [field for field in required_fields if not fields.get(field)]
    if missing:
        return (
            "Ticket draft not created. Ask the user for: "
            f"{', '.join(missing)}."
        )

    try:
        draft = create_jiraticket(**{
            field: fields[field] for field in required_fields
        })
    except ValueError as exc:
        return json.dumps({"status": "error", "message": str(exc)})
    except JiraIntegrationError as exc:
        return json.dumps({"status": "error", "message": str(exc)})
    return json.dumps(draft)


def query_jira_issues_tool(tool_input: str) -> str:
    """Adapt Jira status and requester lookup for the agent's string-input tool."""
    try:
        return SupportTicketToolsEnhanced.retrieve_tickets(tool_input)
    except (ValueError, JiraIntegrationError) as exc:
        return json.dumps({"status": "error", "message": str(exc)})


DIRECTORY_STOPWORDS = {
    "a", "an", "at", "contact", "department", "email", "employee", "employees",
    "find", "for", "from", "in", "is", "number", "of", "on", "phone", "team",
    "the", "who", "works",
}
HOLIDAY_STOPWORDS = {
    "all", "calendar", "company", "date", "day", "federal", "holiday", "holidays",
    "in", "is", "list", "next", "observed", "of", "on", "remaining", "show", "the",
    "upcoming", "us", "what", "when",
}
HOLIDAY_CALENDAR_NOTE = (
    "This is a reference calendar of U.S. federal holidays with weekday observance "
    "dates for California, not a company-specific paid-holiday schedule."
)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file, returning no rows when the file is absent."""
    try:
        with path.open(encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    except FileNotFoundError:
        return []


def _load_policy_sections(policy_dir: Path) -> list[dict[str, str]]:
    """Split Markdown policy files into one knowledge record per headed section."""
    records = []
    for path in sorted(policy_dir.glob("*.md")):
        headings: list[str] = []
        sections: list[tuple[list[str], list[str]]] = [([], [])]
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"(#{1,6})\s+(.+)", line)
            if match:
                headings = headings[:len(match[1]) - 1] + [match[2].strip()]
                sections.append((headings, []))
            else:
                sections[-1][1].append(line)

        for number, (section_headings, lines) in enumerate(sections, start=1):
            body = "\n".join(lines).strip()
            if not body:
                continue
            records.append({
                "id": f"{path.stem}:{number}",
                "subject": " > ".join(section_headings) or path.stem,
                "description": body,
                "priority": "",
                "category": section_headings[0] if section_headings else path.stem,
                "createdAt": "",
                "requesterEmail": "",
                "source": path.name,
                "record_type": "policy",
            })
    return records


def _word_prefix_match(term: str, text: str) -> bool:
    """Return whether a term starts a word, so 'it' matches 'IT' but not 'Smith'."""
    return re.search(rf"(?<!\w){re.escape(term)}", text) is not None


class SupportTicketToolsEnhanced:
    """Enhanced tools with better error handling and small-patch extensions."""
    
    def __init__(self, tickets_path: str | Path | None = None):
        """Load and normalize one JSON file or every JSON file in a data directory."""
        source_path = Path(tickets_path) if tickets_path else PROJECT_ROOT / "data"
        if source_path.is_dir():
            source_files = sorted(
                [*source_path.glob("*.json"), *source_path.glob("*.jsonl")]
            )
        else:
            source_files = [source_path]
        if not source_files:
            raise FileNotFoundError(f"No JSON knowledge files found in {source_path}.")

        self.tickets = []
        self.knowledge_records = []
        for source_file in source_files:
            if source_file.suffix.lower() == ".jsonl":
                records = []
                with source_file.open(encoding="utf-8") as file:
                    for line_number, line in enumerate(file, start=1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"{source_file} line {line_number} is not valid JSON."
                            ) from exc
                        if not isinstance(record, dict):
                            raise ValueError(
                                f"{source_file} line {line_number} must be a JSON object."
                            )
                        messages = record.get("messages")
                        if not isinstance(messages, list):
                            raise ValueError(
                                f"{source_file} line {line_number} must contain "
                                "a 'messages' array."
                            )
                        user_messages = [
                            message.get("content")
                            for message in messages
                            if isinstance(message, dict)
                            and message.get("role") == "user"
                            and isinstance(message.get("content"), str)
                        ]
                        assistant_messages = [
                            message.get("content")
                            for message in messages
                            if isinstance(message, dict)
                            and message.get("role") == "assistant"
                            and isinstance(message.get("content"), str)
                        ]
                        if not user_messages or not assistant_messages:
                            raise ValueError(
                                f"{source_file} line {line_number} must contain "
                                "user and assistant messages with text content."
                            )
                        records.append({
                            "question": user_messages[-1],
                            "answer": assistant_messages[-1],
                            "category": record.get(
                                "category",
                                f"HR Policies - {source_file.stem}",
                            ),
                        })
            else:
                with source_file.open(encoding="utf-8") as file:
                    records = json.load(file)
            if not isinstance(records, list):
                raise ValueError(
                    f"{source_file} must contain a JSON array of records."
                )
            for record_number, record in enumerate(records, start=1):
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{source_file} record {record_number} must be a JSON object."
                    )

                question = record.get("question") or record.get("subject")
                answer = record.get("answer") or record.get("description")
                category = record.get("category")
                if not question or not answer or not category:
                    raise ValueError(
                        f"{source_file} record {record_number} must include "
                        "'question' or 'subject', 'answer' or 'description', "
                        "and 'category'."
                    )

                is_ticket = "id" in record
                normalized_record = {
                    "id": str(record["id"]) if is_ticket else (
                        f"{source_file.stem}:{record_number}"
                    ),
                    "subject": str(question),
                    "description": str(answer),
                    "priority": str(record.get("priority", "")),
                    "category": str(category),
                    "createdAt": str(record.get("createdAt", "")),
                    "requesterEmail": str(record.get("requesterEmail", "")),
                    "source": source_file.name,
                    "record_type": "ticket" if is_ticket else "knowledge",
                }
                self.knowledge_records.append(normalized_record)
                if is_ticket:
                    self.tickets.append(normalized_record)
        
        data_dir = source_path if source_path.is_dir() else PROJECT_ROOT / "data"
        if source_path.is_dir():
            self.knowledge_records.extend(_load_policy_sections(data_dir / "policies"))
        self.directory = _load_csv_rows(data_dir / "company_directory.csv")
        self.holidays = _load_csv_rows(data_dir / "holiday_calendar.csv")

        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        self._setup_vectorstore()
    
    def _setup_vectorstore(self):
        """Build vector and keyword indexes over the ticket documents."""

        documents =[]
        for ticket in self.knowledge_records:
            content_fields = [
                f"SOURCE: {ticket['source']}",
                f"ID: {ticket['id']}",
                f"CATEGORY: {ticket['category']}",
                f"QUESTION OR SUBJECT: {ticket['subject']}",
                f"ANSWER OR DESCRIPTION: {ticket['description']}",
            ]
            if ticket["priority"]:
                content_fields.append(f"PRIORITY: {ticket['priority']}")
            if ticket["createdAt"]:
                content_fields.append(f"CREATEDATE: {ticket['createdAt']}")
            if ticket["requesterEmail"]:
                content_fields.append(
                    f"REQUESTOREMAIL: {ticket['requesterEmail']}"
                )

            doc = LlamaDocument(
                text="\n".join(content_fields),
                metadata={
                    'id': ticket['id'],
                    'subject': ticket['subject'],
                    'description': ticket['description'],
                    'category': ticket['category'],
                    'priority': ticket['priority'],
                    'source': ticket['source'],
                    'record_type': ticket['record_type'],
                }
            )
            documents.append(doc)

        # semantic_splitter = SemanticChunker(
        #     embeddings=self.embeddings,
        #     breakpoint_threshold_type="standard_deviation",
        #     breakpoint_threshold_amount=1.0  # Split when similarity drops 1 std dev below mean
        # )

        # semantic_chunks = semantic_splitter.split_documents(documents)

        # self.vector_store = Chroma.from_documents(
        #     documents=semantic_chunks,
        #     embedding=self.embeddings,
        #     collection_name="semantic_chunks",
        #     collection_metadata={"hnsw:space": "cosine"}
        # )
        chunker = TokenTextSplitter(chunk_size=2048, chunk_overlap=50)
        nodes = chunker.get_nodes_from_documents(documents)
        self.vector_index = VectorStoreIndex(
            nodes,
            embed_model=OpenAIEmbedding(model="text-embedding-3-small")
        )
        self.keyword_index = SimpleKeywordTableIndex(nodes)



    def search_similar_tickets(self, query: str) -> str:
        """Search the normalized knowledge base with exact and hybrid retrieval."""
        if not query or not query.strip():
            return "Error: Please provide a search query."

        normalized_query = re.sub(r"\W+", "", query.casefold())
        exact_matches = [
            record for record in self.knowledge_records
            if re.sub(r"\W+", "", record["subject"].casefold()) == normalized_query
        ]
        if exact_matches:
            return "\n\n".join(
                f"Exact knowledge-base match from {record['source']} "
                f"(Category: {record['category']}):\n"
                f"Question: {record['subject']}\n"
                f"Answer: {record['description']}"
                for record in exact_matches
            )

        vector_results = self.vector_index.as_retriever(
            similarity_top_k=5
        ).retrieve(query)
        keyword_results = self.keyword_index.as_retriever(
            num_chunks_per_query=5
        ).retrieve(query)

        results = self._fuse_rankings((vector_results, keyword_results))
        if not results:
            return "No similar tickets found."
        
        output = "Found related knowledge-base records:\n\n"
        for i, entry in enumerate(results, 1):
            result = entry["result"]
            source = result.metadata.get("source", "Unknown source")
            output += f"--- Result {i} (Source: {source}) ---\n{result.text}\n\n"
        
        return output

    @staticmethod
    def _fuse_rankings(ranked_lists, limit: int = 3) -> list[dict]:
        """Combine semantic and exact-keyword rankings with reciprocal-rank fusion."""
        fused_results = {}
        for ranked_results in ranked_lists:
            seen_ticket_ids = set()
            for rank, result in enumerate(ranked_results, start=1):
                ticket_id = result.metadata.get("id", result.node_id)
                if ticket_id in seen_ticket_ids:
                    continue
                seen_ticket_ids.add(ticket_id)
                entry = fused_results.setdefault(
                    ticket_id,
                    {"score": 0.0, "result": result},
                )
                entry["score"] += 1 / (60 + rank)

        return sorted(
            fused_results.values(),
            key=lambda entry: entry["score"],
            reverse=True,
        )[:limit]

    def search_policies(self, query: str) -> str:
        """Search only the company policy documents, so tickets cannot crowd them out."""
        if not query or not query.strip():
            return "Error: Please provide a search query."

        vector_results = self.vector_index.as_retriever(
            similarity_top_k=5,
            filters=MetadataFilters(
                filters=[ExactMatchFilter(key="record_type", value="policy")]
            ),
        ).retrieve(query)
        keyword_results = [
            result
            for result in self.keyword_index.as_retriever(
                num_chunks_per_query=50
            ).retrieve(query)
            if result.metadata.get("record_type") == "policy"
        ]
        results = self._fuse_rankings((vector_results, keyword_results), limit=4)
        if not results:
            return "No matching policy sections found."

        output = "Found company policy sections:\n\n"
        for i, entry in enumerate(results, 1):
            result = entry["result"]
            output += (
                f"--- Result {i} (Source: {result.metadata.get('source')}, "
                f"Section: {result.metadata.get('subject')}) ---\n{result.text}\n\n"
            )
        return output

    def search_company_directory(self, query: str) -> str:
        """Find employees by name, ID, department, job title, email, or phone."""
        if not self.directory:
            return "The company directory is not available."

        terms = [
            term.strip(".,?!'\"")
            for term in re.findall(r"[\w@.+'-]+", (query or "").casefold())
            if term.strip(".,?!'\"") not in DIRECTORY_STOPWORDS
        ]
        if not terms:
            return "Please provide a name, employee ID, department, or job title."

        def searchable(employee: dict[str, str]) -> str:
            return " ".join(
                value for key, value in employee.items() if key != "manager_id"
            ).casefold()

        matches = [
            employee for employee in self.directory
            if all(_word_prefix_match(term, searchable(employee)) for term in terms)
        ]
        if not matches:
            return f"No directory entries found for: {query}"

        names_by_id = {e["employee_id"]: e["name"] for e in self.directory}
        output = [f"Found {len(matches)} directory entries (fictional sample data):"]
        for employee in matches[:15]:
            manager_id = employee["manager_id"]
            manager = (
                f"{names_by_id.get(manager_id, 'Unknown')} ({manager_id})"
                if manager_id else "None (top-level)"
            )
            output.append(
                f"- {employee['name']} ({employee['employee_id']}) | "
                f"{employee['department']} | {employee['job_title']} | "
                f"{employee['email']} | {employee['phone_number']} | "
                f"Manager: {manager}"
            )
        if len(matches) > 15:
            output.append(f"...and {len(matches) - 15} more. Ask for a narrower search.")
        if len(matches) == 1:
            reports = [
                f"{e['name']} ({e['employee_id']})" for e in self.directory
                if e["manager_id"] == matches[0]["employee_id"]
            ]
            output.append("Direct reports: " + (", ".join(reports) or "None"))
        return "\n".join(output)

    def search_holiday_calendar(self, query: str) -> str:
        """Look up holidays by name, year, or month, or find upcoming holidays."""
        if not self.holidays:
            return "The holiday calendar is not available."

        today = date.today()
        text = (query or "").casefold()
        years = {int(year) for year in re.findall(r"\b20\d{2}\b", text)}
        if "this year" in text:
            years.add(today.year)
        if "next year" in text:
            years.add(today.year + 1)
        text = text.replace("this year", " ").replace("next year", " ")

        month_names = {
            name.casefold(): number
            for number, name in enumerate(calendar.month_name) if name
        }
        words = re.findall(r"[a-z]+", text)
        months = {month_names[word] for word in words if word in month_names}
        name_terms = [
            word for word in words
            if word not in month_names and word not in HOLIDAY_STOPWORDS
        ]
        upcoming = bool(re.search(r"\b(next|upcoming|remaining)\b", text))

        rows = sorted(
            (
                {**row, "_observed": date.fromisoformat(row["observed_date"])}
                for row in self.holidays
            ),
            key=lambda row: row["_observed"],
        )
        if not (years or months or name_terms or upcoming):
            years = {today.year}
        if years:
            rows = [r for r in rows if r["_observed"].year in years]
        if months:
            rows = [r for r in rows if r["_observed"].month in months]
        if upcoming:
            rows = [r for r in rows if r["_observed"] >= today]

        note = ""
        if name_terms:
            named = [
                row for row in rows
                if all(
                    _word_prefix_match(term, row["holiday"].casefold())
                    for term in name_terms
                )
            ]
            if named:
                rows = named
            else:
                note = "No holiday name matched exactly; showing the available dates.\n"
        elif upcoming and not re.search(r"\b(all|remaining)\b", text):
            rows = rows[:3]

        header = f"Today is {today:%A, %Y-%m-%d}. {HOLIDAY_CALENDAR_NOTE}\n{note}"
        if not rows:
            return header + "No matching holidays found in the calendar."
        return header + "\n".join(
            f"- {row['holiday']}: {date.fromisoformat(row['date']):%A, %Y-%m-%d}; "
            f"observed {row['_observed']:%A, %Y-%m-%d}"
            for row in rows
        )

    def search_ticket_by_id(self, query: str) -> str:
        """Retrieve one local helpdesk ticket by its exact ID from the vector index."""
        match = re.search(r"\b[a-zA-Z0-9]{8,}\b", query or "")
        if not match:
            return "Please provide the exact helpdesk ticket ID."

        ticket_id = match.group(0)
        results = self.vector_index.as_retriever(
            similarity_top_k=1,
            filters=MetadataFilters(
                filters=[ExactMatchFilter(key="id", value=ticket_id)]
            ),
        ).retrieve(f"ID: {ticket_id}")
        if not results:
            return f"No helpdesk ticket found with ID {ticket_id}."

        result = results[0]
        return f"Helpdesk ticket (ID: {ticket_id}):\n{result.text}"

    @staticmethod
    def retrieve_tickets(query: str) -> str:
        """Look up one Jira issue by key or list project issues for a requester."""
        query = query.strip()
        base_url = os.getenv("JIRA_BASE_URL", "").strip().rstrip("/")
        jira_email = os.getenv("JIRA_EMAIL", "").strip()
        api_token = os.getenv("JIRA_API_TOKEN", "").strip()
        project_key = os.getenv("JIRA_PROJECT_KEY", "").strip()
        missing_config = [
            name for name, value in (
                ("JIRA_BASE_URL", base_url),
                ("JIRA_EMAIL", jira_email),
                ("JIRA_API_TOKEN", api_token),
                ("JIRA_PROJECT_KEY", project_key),
            )
            if not value
        ]
        if missing_config:
            raise JiraIntegrationError(
                "Configure the following environment variables before searching "
                f"Jira: {', '.join(missing_config)}."
            )

        parsed_url = urlsplit(base_url)
        if (
            parsed_url.scheme != "https"
            or not parsed_url.netloc
            or parsed_url.username
            or parsed_url.password
            or parsed_url.query
            or parsed_url.fragment
            or parsed_url.path not in ("", "/")
        ):
            raise JiraIntegrationError(
                "JIRA_BASE_URL must be an HTTPS Jira site URL without credentials, "
                "a path, query, or fragment."
            )
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", project_key):
            raise JiraIntegrationError("JIRA_PROJECT_KEY is not a valid Jira project key.")

        credentials = base64.b64encode(
            f"{jira_email}:{api_token}".encode("utf-8")
        ).decode("ascii")
        issue_key_match = re.search(r"\b([A-Z][A-Z0-9_]*-\d+)\b", query, re.IGNORECASE)
        if issue_key_match:
            issue_key = issue_key_match.group(1).upper()
            if not re.fullmatch(
                rf"{re.escape(project_key)}-\d+",
                issue_key,
                re.IGNORECASE,
            ):
                return (
                    f"Jira issue {issue_key} is not in the configured project "
                    f"{project_key}."
                )
            request = Request(
                f"{base_url}/rest/api/3/issue/{quote(issue_key, safe='-')}?fields=key,summary,status",
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Accept": "application/json",
                },
                method="GET",
            )
            try:
                with urlopen(request, timeout=20) as response:
                    issue = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code == 404:
                    return (
                        f"Jira issue {issue_key} was not found or is not visible "
                        "to the configured Jira account."
                    )
                error_body = exc.read().decode("utf-8", errors="replace")
                try:
                    details = json.loads(error_body)
                    error_message = "; ".join(
                        details.get("errorMessages", [])
                        + list(details.get("errors", {}).values())
                    )
                except (json.JSONDecodeError, AttributeError):
                    error_message = error_body[:1000]
                message = error_message or f"Jira returned HTTP {exc.code}."
                raise JiraIntegrationError(
                    f"Jira could not retrieve issue {issue_key}: {message}"
                ) from exc
            except URLError as exc:
                raise JiraIntegrationError(
                    f"Could not connect to Jira: {exc.reason}"
                ) from exc
            except (TimeoutError, OSError) as exc:
                raise JiraIntegrationError(
                    f"Could not complete the Jira request: {exc}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise JiraIntegrationError(
                    "Jira returned an invalid response while retrieving the issue."
                ) from exc

            fields = issue.get("fields", {})
            status = fields.get("status") or {}
            return (
                f"Jira issue {issue.get('key', issue_key)}: "
                f"{fields.get('summary', 'No summary')} "
                f"(Status: {status.get('name', 'Unknown')}) "
                f"{base_url}/browse/{issue.get('key', issue_key)}"
            )

        email = query
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            raise ValueError(
                "Provide your requester work email to list your Jira issues, "
                "or a Jira issue key such as HELP-123 to check one issue."
            )

        escaped_email = email.replace("\\", "\\\\").replace('"', '\\"')
        jql = (
            f'project = "{project_key}" AND description ~ "\\"{escaped_email}\\"" '
            "ORDER BY created DESC"
        )
        issues = []
        next_page_token = None

        while True:
            query_params = (
                f"jql={quote(jql, safe='')}"
                "&fields=key,summary,status"
                "&maxResults=100"
            )
            if next_page_token:
                query_params += f"&nextPageToken={quote(next_page_token, safe='')}"
            request = Request(
                f"{base_url}/rest/api/3/search/jql?{query_params}",
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Accept": "application/json",
                },
                method="GET",
            )
            try:
                with urlopen(request, timeout=20) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                try:
                    details = json.loads(error_body)
                    error_message = "; ".join(
                        details.get("errorMessages", [])
                        + list(details.get("errors", {}).values())
                    )
                except (json.JSONDecodeError, AttributeError):
                    error_message = error_body[:1000]
                message = error_message or f"Jira returned HTTP {exc.code}."
                raise JiraIntegrationError(
                    f"Jira could not retrieve issues: {message}"
                ) from exc
            except URLError as exc:
                raise JiraIntegrationError(
                    f"Could not connect to Jira: {exc.reason}"
                ) from exc
            except (TimeoutError, OSError) as exc:
                raise JiraIntegrationError(
                    f"Could not complete the Jira request: {exc}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise JiraIntegrationError(
                    "Jira returned an invalid response while searching issues."
                ) from exc

            issues.extend(response_data.get("issues", []))
            next_page_token = response_data.get("nextPageToken")
            if not next_page_token:
                break

        if not issues:
            return f"No Jira tickets found for {email} in project {project_key}."

        lines = [f"Jira tickets for {email}:"]
        for issue in issues:
            fields = issue.get("fields", {})
            status = fields.get("status") or {}
            lines.append(
                f"- {issue.get('key', 'Unknown')}: "
                f"{fields.get('summary', 'No summary')} "
                f"(Status: {status.get('name', 'Unknown')}) "
                f"{base_url}/browse/{issue.get('key', '')}"
            )
        return "\n".join(lines)

    def get_tools(self):
        """Get all tools with improved descriptions."""
        return [
            Tool(
                name="SearchCompanyPolicies",
                func=self.search_policies,
                description="""Search the official company policy documents: HR, leave, payroll and
compensation, onboarding, IT support, and VPN setup.
Use FIRST for any question about company rules, entitlements, or procedures.
Input: The user's question."""
            ),
            Tool(
                name="SearchCompanyDirectory",
                func=self.search_company_directory,
                description="""Look up employees in the company directory, including email, phone,
department, job title, manager, and direct reports.
Use for questions like "who is X", "who manages X", or "who works in IT Support".
Input: Only a name, employee ID, department, job title, email, or phone number."""
            ),
            Tool(
                name="SearchHolidayCalendar",
                func=self.search_holiday_calendar,
                description="""Look up dates in the holiday calendar and find upcoming holidays.
Use for questions like "when is Labor Day" or "what is the next holiday".
Input: A holiday name, year, or month, or "next holiday", "upcoming holidays",
or "all holidays 2027"."""
            ),
            Tool(
                name="SearchSimilarTickets",
                func=self.search_similar_tickets,
                description="""Search the combined SmartHR knowledge base, including helpdesk tickets,
HR policy Q&A, and IT support FAQs. Use for questions about policies, procedures,
troubleshooting, or related past issues.
Input: The user's question or a clear description of the problem.
This is the PRIMARY tool for knowledge questions."""
            ),
            Tool(
                name="SearchTicketStatus",
                func=self.search_ticket_by_id,
                description="""Retrieve one ticket from the local helpdesk ticket database by exact ticket ID.
Use when the user mentions a specific helpdesk ticket ID, for example "Show me ticket v4f7z7h45".
Input: The exact ticket ID. This searches the local vector index, not Jira."""
            ),
            Tool(
                name="QueryJiraIssues",
                func=query_jira_issues_tool,
                description="""Check Jira issue status or list Jira tickets created for a requester.
Use for Jira-specific requests, not local helpdesk ticket IDs.
Input: A Jira issue key such as HELP-123 to return its status, or the requester's
work email to list their issues in the configured Jira project. If neither is
available, ask the user for the issue key or work email."""
            ),
            Tool(
                name="create_jiraticket",
                func=create_jiraticket_tool,
                description="""Create a support ticket in the configured Jira project.
Use only when the user explicitly confirms they want to submit a support ticket
and you have collected all five required details: email, summary, description,
category, and priority. If any detail is missing, ask the user for it before
calling this tool. Tell the user the returned Jira issue key after successful creation.
Input must be a JSON object encoded as a string with exactly these fields:
{"email":"...","summary":"...","description":"...","category":"...","priority":"..."}.
This tool creates a real Jira issue; do not call it without explicit user consent."""
            )
        ]


retrieve_jiratickets = SupportTicketToolsEnhanced.retrieve_tickets


tool_manager = None
tools = []
llm_with_tools = None


def initialize_agent():
    """Load tickets and initialize the embedding index and chat model on demand."""
    global tool_manager, tools, llm_with_tools

    if llm_with_tools is not None:
        return llm_with_tools, tools

    manager = SupportTicketToolsEnhanced()
    available_tools = manager.get_tools()
    llm = ChatOpenAI(model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'), temperature=0)
    tool_definitions = []
    for tool in available_tools:
        if tool.name == "create_jiraticket":
            parameters = {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The requester's work email address.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "A concise title for the Jira issue.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed problem description and steps tried.",
                    },
                    "category": {
                        "type": "string",
                        "description": "The support category for the issue.",
                    },
                    "priority": {
                        "type": "string",
                        "description": "One of the configured support priority values.",
                    },
                },
                "required": [
                    "email",
                    "summary",
                    "description",
                    "category",
                    "priority",
                ],
            }
        else:
            parameters = {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The input to the tool",
                    }
                },
                "required": ["input"],
            }
        tool_definitions.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        })

    tool_manager = manager
    tools = available_tools
    llm_with_tools = llm.bind(tools=tool_definitions)
    return llm_with_tools, tools


def get_ticket_form_options() -> dict[str, list[str]]:
    """Return category and priority choices from the loaded ticket dataset."""
    tickets = (
        tool_manager.tickets
        if tool_manager is not None
        else json.loads(
            (PROJECT_ROOT / "data" / "IT-helpdesktickets.json").read_text(
                encoding="utf-8"
            )
        )
    )
    return {
        "categories": sorted({ticket["category"] for ticket in tickets}),
        "priorities": sorted({
            ticket["priority"] for ticket in tickets if ticket.get("priority")
        }),
    }

# ============================================================================
# Exercise 6: Custom Agent Prompt
# ============================================================================
CUSTOM_SYSTEM_PROMPT = """You are a SmartHR assistant with access to company policy documents, the company directory, the holiday calendar, and a searchable knowledge base containing helpdesk tickets, HR policy Q&A, and IT support FAQs.

ALWAYS follow these rules:
1. State which tool you're about to use before calling it
2. If the query is ambiguous, ask a clarifying question FIRST
3. Answer from the retrieved knowledge-base information; do not invent facts or fixes.
4. If retrieved information does not answer the user's question, begin your final response with the exact marker [NO_ANSWER], then say you do not have enough information.
5. When you can answer, rate your confidence (High/Medium/Low), cite the available source or ticket ID when useful, and suggest one related topic.
6. Search every relevant source before using [NO_ANSWER], and use more than one tool when a question spans sources. If policy documents and other records disagree, prefer the policy documents and mention the conflict.

Your tools:
- SearchCompanyPolicies: First choice for company rules, entitlements, and procedures
- SearchCompanyDirectory: For finding employees, contact details, managers, and direct reports
- SearchHolidayCalendar: For holiday dates and upcoming holidays
- SearchSimilarTickets: For questions about HR policies, IT support FAQs, troubleshooting, and helpdesk records
- SearchTicketStatus: For looking up a specific helpdesk ticket by ID in the local vector database
- QueryJiraIssues: For checking a Jira issue key's status or listing Jira issues by requester work email
- create_jiraticket: Submit a Jira issue only after the user explicitly confirms and provides all five required details. Ask for any missing details before calling it. After creation, report the issue key.
"""


def run_agent(
    query: str,
    max_iterations: int = 5,
    custom_prompt: str = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict:
    """
    Run the agent with tracking for evaluation.
    Returns dict with response, tools_used, and iterations.
    """
    if llm_with_tools is None:
        initialize_agent()

    # Allow callers to swap prompts for experimentation while keeping a safe default.
    system_prompt = custom_prompt or CUSTOM_SYSTEM_PROMPT
    
    messages = [SystemMessage(content=system_prompt)]
    for turn in (conversation_history or [])[-10:]:
        role = turn.get("role")
        content = turn.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=query))
    
    # Track tool trajectory for later evaluation/debugging.
    tools_used = []
    
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Final answer path: model returned plain content with no tool requests.
        if not response.tool_calls:
            response_text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            return {
                'response': response_text,
                'needs_ticket': response_text.lstrip().startswith("[NO_ANSWER]"),
                'tools_used': tools_used,
                'iterations': i + 1
            }
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            if tool_name == "create_jiraticket":
                tool_input = json.dumps(tool_args)
            else:
                tool_input = tool_args.get("input", "")

            print(f"🔧 Tool: {tool_name} | Input: {tool_input[:50]}...")
            tools_used.append(tool_name)
            
            # Resolve and execute requested tool safely via name lookup.
            tool_output = None
            for tool in tools:
                if tool.name == tool_name:
                    tool_output = tool.func(tool_input)
                    break
            
            if tool_output is None:
                tool_output = f"Error: Tool {tool_name} not found"
            
            # Feed tool output back into model context for the next reasoning step.
            messages.append(ToolMessage(
                content=tool_output,
                tool_call_id=tool_call["id"]
            ))
    
    return {
        'response': "Maximum iterations reached.",
        'needs_ticket': False,
        'tools_used': tools_used,
        'iterations': max_iterations
    }

def run_demo():
    """Run the original sample queries when this module is executed directly."""
    test_queries = [
        ("Where is India", "SearchSimilarTickets"),
        ("Show me ticket v4f7z7h45", "SearchSimilarTickets"),
        ("Access issue", "SearchSimilarTickets"),
        ("email issue", "SearchSimilarTickets"),
    ]

    for query, expected_tool in test_queries:
        print(f"\n--- Query: '{query}' ---")
        print(f"Expected tool: {expected_tool}")
        result = run_agent(query)
        print(f"Actual tools used: {result['tools_used']}")
        print(f"Iterations: {result['iterations']}")
        correct = expected_tool in result['tools_used']
        print(f"Correct: {'✓' if correct else '✗'}")
        print(f"Response: {result['response'][:150]}...")


if __name__ == "__main__":
    print("EXERCISE 1 & 2: Test Different Queries")
    run_demo()
