"""Streamlit chat interface for the SmartHR IT support agent."""

import csv
import json
from datetime import date
from pathlib import Path

import streamlit as st

from smartHR import (
    JiraIntegrationError,
    create_jiraticket,
    get_ticket_form_options,
    initialize_agent,
    retrieve_jiratickets,
    run_agent,
)
from smartHR_auth import (
    AuthConfigurationError,
    authenticate,
    create_user,
    ensure_initial_admin,
    get_active_user,
    initialize_auth,
    list_users,
    reset_password,
    set_user_active,
)


st.set_page_config(
    page_title="SmartHR Support",
    page_icon="💬",
    layout="centered",
)

st.title("SmartHR Support")

try:
    initialize_auth()
    ensure_initial_admin()
except AuthConfigurationError as exc:
    st.error(str(exc))
    st.stop()

if not list_users():
    st.info("SmartHR login is enabled, but the initial admin account is not configured.")
    st.markdown(
        "Set `SMART_HR_ADMIN_USERNAME` and `SMART_HR_ADMIN_PASSWORD` in the "
        "environment (or local `.env` file), then restart the app. Use a unique "
        "password with at least 12 characters."
    )
    st.stop()

if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None

if st.session_state.authenticated_user is None:
    st.subheader("Sign in")
    with st.form("login_form"):
        login_username = st.text_input("Username", max_chars=32)
        login_password = st.text_input(
            "Password",
            type="password",
            max_chars=256,
        )
        login_submitted = st.form_submit_button("Sign in")

    if login_submitted:
        user = authenticate(login_username, login_password)
        if user:
            st.session_state.authenticated_user = user
            st.rerun()
        st.error("Sign-in failed. Check your credentials or contact an admin.")
    st.stop()

session_user = st.session_state.authenticated_user
authenticated_user = get_active_user(session_user["id"])
if (
    authenticated_user is None
    or authenticated_user["session_version"] != session_user["session_version"]
):
    st.session_state.authenticated_user = None
    st.warning("Your session is no longer valid. Please sign in again.")
    st.stop()
st.session_state.authenticated_user = authenticated_user

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_ticket_request" not in st.session_state:
    st.session_state.pending_ticket_request = None
if "ticket_form_open" not in st.session_state:
    st.session_state.ticket_form_open = False
if "ticket_draft" not in st.session_state:
    st.session_state.ticket_draft = None
if "jira_issue_result" not in st.session_state:
    st.session_state.jira_issue_result = None
if "jira_query_results" not in st.session_state:
    st.session_state.jira_query_results = None

with st.sidebar:
    st.caption(f"Signed in as **{authenticated_user['username']}**")
    if st.button("Sign out", use_container_width=True):
        for key in (
            "authenticated_user",
            "messages",
            "pending_ticket_request",
            "ticket_form_open",
            "ticket_draft",
            "jira_issue_result",
            "jira_query_results",
            "employment_verification_letter",
            "demo_employment_verification_letter",
            "demo_employment_verification_filename",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    st.header("Navigation")
    pages = [
        "Support Chat",
        "Log Jira Issue",
        "Query Jira Issues",
        "Company Directory",
        "Holiday Calendar",
        "Policy Documents",
        "Employment Verification",
    ]
    if authenticated_user["role"] == "admin":
        pages.extend(["HR Employment Letter", "Manage Users"])
    active_page = st.radio(
        "Choose a page",
        pages,
        label_visibility="collapsed",
    )
    st.header("About")
    st.write(
        "SmartHR Support uses the company helpdesk ticket dataset to find similar "
        "issues and suggest relevant troubleshooting information."
    )
    st.caption("Do not enter passwords, MFA codes, or other sensitive information.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_ticket_request = None
        st.session_state.ticket_form_open = False
        st.session_state.ticket_draft = None
        st.session_state.jira_issue_result = None
        st.session_state.jira_query_results = None
        st.rerun()

if active_page == "Support Chat":
    st.caption("Ask about HR policies, IT support FAQs, or helpdesk issues.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("How can I help?")
    if question:
        st.session_state.pending_ticket_request = None
        st.session_state.ticket_form_open = False
        st.session_state.ticket_draft = None
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching company sources and preparing an answer..."):
                    initialize_agent()
                    result = run_agent(
                        question,
                        conversation_history=st.session_state.messages[:-1],
                    )
                answer = str(result["response"]).removeprefix("[NO_ANSWER]").strip()
                st.markdown(answer)
                if result.get("tools_used"):
                    st.caption("Tools used: " + ", ".join(result["tools_used"]))
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                if result.get("needs_ticket"):
                    st.session_state.pending_ticket_request = question
                    st.session_state.ticket_form_open = False
                    st.session_state.ticket_draft = None
            except Exception as exc:
                st.error("The support assistant could not complete this request.")
                with st.expander("Error details"):
                    st.code(str(exc))

    if st.session_state.pending_ticket_request:
        st.info(
            "I couldn't find a reliable answer in the SmartHR knowledge base. "
            "You can prepare a Jira ticket if you'd like."
        )
        if not st.session_state.ticket_form_open:
            if st.button("Prepare Jira ticket"):
                st.session_state.ticket_form_open = True
                st.rerun()

        if st.session_state.ticket_form_open:
            st.subheader("Submit a support ticket to Jira")
            st.caption("Review the details before submission. Submitting creates a Jira issue.")
            try:
                options = get_ticket_form_options()
            except Exception as exc:
                st.error("Could not load ticket categories and priorities.")
                st.code(str(exc))
            else:
                original_question = st.session_state.pending_ticket_request
                with st.form("support_ticket_draft"):
                    email = st.text_input("Your work email")
                    summary = st.text_input(
                        "Summary",
                        value=original_question[:120],
                        max_chars=120,
                    )
                    description = st.text_area(
                        "Description",
                        value=f"Original question: {original_question}\n\nAdditional details: ",
                        height=160,
                    )
                    category = st.selectbox("Category", options["categories"])
                    priority = st.selectbox(
                        "Priority",
                        options["priorities"],
                        index=(
                            options["priorities"].index("Medium")
                            if "Medium" in options["priorities"]
                            else 0
                        ),
                    )
                    create_draft = st.form_submit_button("Submit ticket to Jira")

                if create_draft:
                    try:
                        st.session_state.ticket_draft = create_jiraticket(
                            email=email,
                            summary=summary,
                            description=description,
                            category=category,
                            priority=priority,
                        )
                        st.session_state.pending_ticket_request = None
                        st.session_state.ticket_form_open = False
                    except ValueError as exc:
                        st.error(str(exc))
                    except JiraIntegrationError as exc:
                        st.error(f"Could not submit the Jira issue: {exc}")

    if st.session_state.ticket_draft:
        issue = st.session_state.ticket_draft["issue"]
        st.success(f"Jira issue created: {issue['key']}")
        st.markdown(f"[Open {issue['key']} in Jira]({issue['url']})")

elif active_page == "Log Jira Issue":
    st.subheader("Log a Jira Issue")
    st.caption("Complete the form to create a support issue in the configured Jira project.")
    try:
        options = get_ticket_form_options()
    except Exception as exc:
        st.error("Could not load ticket categories and priorities.")
        st.code(str(exc))
    else:
        with st.form("log_jira_issue"):
            email = st.text_input("Your work email")
            summary = st.text_input("Summary", max_chars=120)
            description = st.text_area("Description", height=180)
            category = st.selectbox("Category", options["categories"])
            priority = st.selectbox(
                "Priority",
                options["priorities"],
                index=(
                    options["priorities"].index("Medium")
                    if "Medium" in options["priorities"]
                    else 0
                ),
            )
            submit_issue = st.form_submit_button("Create Jira issue")

        if submit_issue:
            try:
                st.session_state.jira_issue_result = create_jiraticket(
                    email=email,
                    summary=summary,
                    description=description,
                    category=category,
                    priority=priority,
                )
            except ValueError as exc:
                st.error(str(exc))
            except JiraIntegrationError as exc:
                st.error(f"Could not submit the Jira issue: {exc}")

        if st.session_state.jira_issue_result:
            issue = st.session_state.jira_issue_result["issue"]
            st.success(f"Jira issue created: {issue['key']}")
            st.markdown(f"[Open {issue['key']} in Jira]({issue['url']})")

elif active_page == "Query Jira Issues":
    st.subheader("Query Jira Issues")
    st.caption("Find support issues by the work email included when the ticket was created.")
    with st.form("query_jira_issues"):
        requester_email = st.text_input("Requester work email")
        query_issues = st.form_submit_button("Find my Jira issues")

    if query_issues:
        try:
            st.session_state.jira_query_results = retrieve_jiratickets(requester_email)
        except ValueError as exc:
            st.error(str(exc))
        except JiraIntegrationError as exc:
            st.error(f"Could not retrieve Jira issues: {exc}")

    if st.session_state.jira_query_results:
        st.markdown(st.session_state.jira_query_results)

elif active_page == "Company Directory":
    st.subheader("Company Directory")
    st.warning(
        "This is fictional sample data for demonstration; it does not represent "
        "actual SmartHR employees."
    )
    directory_path = Path(__file__).resolve().parents[1] / "data" / "company_directory.csv"
    try:
        with directory_path.open(encoding="utf-8", newline="") as directory_file:
            employees = list(csv.DictReader(directory_file))
    except OSError as exc:
        st.error(f"Could not load the company directory: {exc}")
    else:
        directory_tab, org_chart_tab = st.tabs(["Directory", "Org Chart"])

        with directory_tab:
            departments = ["All departments"] + sorted({
                employee["department"] for employee in employees
            })
            selected_department = st.selectbox("Department", departments)
            search_term = st.text_input(
                "Search directory",
                placeholder=(
                    "Search by name, employee ID, title, department, email, or phone"
                ),
            ).strip().casefold()

            filtered_employees = [
                employee for employee in employees
                if (
                    selected_department == "All departments"
                    or employee["department"] == selected_department
                )
                and (
                    not search_term
                    or any(
                        search_term in value.casefold()
                        for value in employee.values()
                    )
                )
            ]
            st.caption(
                f"Showing {len(filtered_employees)} of {len(employees)} sample employees"
            )
            st.dataframe(
                filtered_employees,
                hide_index=True,
                use_container_width=True,
            )

        with org_chart_tab:
            executives = [
                employee for employee in employees
                if not employee["manager_id"]
            ]
            if not executives:
                st.error("The directory has no top-level employee for the org chart.")
            else:
                graph_lines = [
                    "digraph CompanyOrgChart {",
                    '  graph [rankdir="LR", ordering="out", nodesep="0.2", ranksep="0.4"];',
                    '  node [shape="box", style="rounded,filled", fillcolor="#EAF2F8", fontsize="9", margin="0.08,0.04"];',
                    '  edge [arrowsize="0.7"];',
                ]
                for employee in employees:
                    label = (
                        f"{employee['name']}\n{employee['job_title']}\n"
                        f"{employee['employee_id']}"
                    )
                    graph_lines.append(
                        f"  {json.dumps(employee['employee_id'])} "
                        f"[label={json.dumps(label, ensure_ascii=False)}];"
                    )
                    if employee["manager_id"]:
                        graph_lines.append(
                            f"  {json.dumps(employee['manager_id'])} -> "
                            f"{json.dumps(employee['employee_id'])};"
                        )
                graph_lines.append("}")
                st.graphviz_chart(
                    "\n".join(graph_lines),
                    use_container_width=True,
                )

elif active_page == "Holiday Calendar":
    st.subheader("Holiday Calendar")
    st.caption(
        "U.S. federal holidays with weekday observance dates for California. "
        "This is a reference calendar, not a company-specific paid-holiday schedule."
    )
    holiday_path = Path(__file__).resolve().parents[1] / "data" / "holiday_calendar.csv"
    try:
        with holiday_path.open(encoding="utf-8", newline="") as holiday_file:
            holidays = list(csv.DictReader(holiday_file))
    except OSError as exc:
        st.error(f"Could not load the holiday calendar: {exc}")
    else:
        available_years = sorted({holiday["year"] for holiday in holidays})
        selected_year = st.selectbox(
            "Calendar year",
            available_years,
            index=available_years.index("2026") if "2026" in available_years else 0,
        )
        year_holidays = [
            holiday for holiday in holidays if holiday["year"] == selected_year
        ]
        st.dataframe(
            [
                {
                    "Holiday": holiday["holiday"],
                    "Date": holiday["date"],
                    "Weekday observed": holiday["observed_date"],
                }
                for holiday in year_holidays
            ],
            hide_index=True,
            use_container_width=True,
        )

elif active_page == "Policy Documents":
    st.subheader("Policy Documents")
    st.caption("Browse the available HR, IT, onboarding, and payroll policy documents.")
    policy_dir = Path(__file__).resolve().parents[1] / "data" / "policies"
    policy_files = sorted(policy_dir.glob("*.md"))

    if not policy_files:
        st.info("No policy documents are available.")
    else:
        for policy_file in policy_files:
            try:
                policy_content = policy_file.read_text(encoding="utf-8")
            except OSError as exc:
                st.error(f"Could not load {policy_file.name}: {exc}")
                continue

            st.markdown(policy_content)
            st.divider()

elif (
    active_page == "HR Employment Letter"
    and authenticated_user["role"] == "admin"
):
    st.subheader("HR Employment Verification Letter")
    st.warning(
        "Admin-only tool. The company directory contains fictional sample data "
        "and is not used to populate this letter. Enter only HR-verified details."
    )
    st.caption(
        "Generated letters are not saved to the application database. Verify "
        "the facts and confirm you are authorized before downloading or sharing."
    )

    with st.form("employment_verification_form"):
        company_name = st.text_input(
            "Company name",
            value="SmartHR Solutions",
            max_chars=120,
        )
        employee_name = st.text_input("Employee full name", max_chars=120)
        job_title = st.text_input("Job title", max_chars=120)
        department = st.text_input("Department", max_chars=120)
        employment_type = st.selectbox(
            "Employment type",
            ("Full-time", "Part-time", "Contract", "Temporary", "Other"),
        )
        employment_status = st.selectbox(
            "Employment status",
            ("Currently employed", "Former employee"),
        )
        employment_start_date = st.date_input(
            "Employment start date",
            value=date.today(),
        )
        employment_end_date = None
        if employment_status == "Former employee":
            employment_end_date = st.date_input(
                "Employment end date",
                value=date.today(),
            )
        work_location = st.text_input("Work location (optional)", max_chars=120)
        recipient = st.text_input(
            "Recipient organization or person (optional)",
            max_chars=160,
        )
        purpose = st.text_input(
            "Verification purpose (optional)",
            max_chars=160,
        )
        compensation = st.text_input(
            "Compensation (optional; include only when authorized)",
            max_chars=120,
        )
        signer_name = st.text_input("HR representative name", max_chars=120)
        signer_title = st.text_input("HR representative title", max_chars=120)
        signer_email = st.text_input("HR contact email (optional)", max_chars=254)
        verified_and_authorized = st.checkbox(
            "I confirm these details are HR-verified and I am authorized to "
            "issue this letter."
        )
        generate_letter = st.form_submit_button("Generate letter")

    if generate_letter:
        st.session_state.pop("employment_verification_letter", None)
        missing_fields = [
            label
            for label, value in (
                ("Company name", company_name),
                ("Employee full name", employee_name),
                ("Job title", job_title),
                ("Department", department),
                ("HR representative name", signer_name),
                ("HR representative title", signer_title),
            )
            if not value.strip()
        ]
        if missing_fields:
            st.error("Please complete: " + ", ".join(missing_fields) + ".")
        elif (
            employment_end_date is not None
            and employment_end_date < employment_start_date
        ):
            st.error("Employment end date cannot be before the start date.")
        elif employment_start_date > date.today():
            st.error("Employment start date cannot be in the future.")
        elif (
            employment_end_date is not None
            and employment_end_date > date.today()
        ):
            st.error("Employment end date cannot be in the future.")
        elif not verified_and_authorized:
            st.error(
                "Confirm that the details are HR-verified and you are authorized "
                "to issue this letter."
            )
        else:
            employment_period = (
                f"from {employment_start_date.isoformat()} to "
                f"{employment_end_date.isoformat()}"
                if employment_end_date is not None
                else f"since {employment_start_date.isoformat()}"
            )
            recipient_line = (
                f"To: {recipient.strip()}\n\n"
                if recipient.strip()
                else ""
            )
            letter_lines = [
                company_name.strip(),
                "Employment Verification",
                f"Date: {date.today().isoformat()}",
                "",
                recipient_line.rstrip(),
                "To Whom It May Concern,"
                if not recipient.strip()
                else "Dear Verification Recipient,",
                "",
                (
                    f"This letter confirms that {employee_name.strip()} "
                    f"{'is currently employed' if employment_end_date is None else 'was employed'} "
                    f"by {company_name.strip()} as a "
                    f"{job_title.strip()} in the {department.strip()} department "
                    f"({employment_type.lower()}) {employment_period}."
                ),
                f"Employment status: {employment_status}.",
            ]
            if work_location.strip():
                letter_lines.append(f"Work location: {work_location.strip()}.")
            if purpose.strip():
                letter_lines.append(f"Purpose of verification: {purpose.strip()}.")
            if compensation.strip():
                letter_lines.append(
                    f"Compensation: {compensation.strip()} "
                    "(included at the employer's authorized disclosure)."
                )
            letter_lines.extend([
                "",
                "This letter is provided for employment verification only. "
                "Please contact the HR representative below if additional "
                "verification is required.",
                "",
                "Sincerely,",
                signer_name.strip(),
                signer_title.strip(),
            ])
            if signer_email.strip():
                letter_lines.append(signer_email.strip())

            st.session_state.employment_verification_letter = "\n".join(
                line for line in letter_lines if line is not None
            )

    if st.session_state.get("employment_verification_letter"):
        st.text_area(
            "Letter preview",
            value=st.session_state.employment_verification_letter,
            height=360,
            disabled=True,
        )
        st.download_button(
            "Download employment verification letter",
            data=st.session_state.employment_verification_letter,
            file_name="employment_verification_letter.txt",
            mime="text/plain",
        )

elif active_page == "Employment Verification":
    st.subheader("Employment Verification Letter — Test")
    st.warning(
        "TEST DATA ONLY — These 100 fictional employee records are for "
        "demonstrating the self-service workflow. Letters generated here are "
        "not proof of employment and must not be used for official purposes."
    )
    is_admin = authenticated_user["role"] == "admin"
    if is_admin:
        st.caption("As an admin, you can generate a sample letter for any employee.")
    else:
        st.caption("You can generate a sample letter only for the employee account linked to your username.")
    test_data_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "employment_verification_testdata.csv"
    )
    try:
        with test_data_path.open(encoding="utf-8", newline="") as test_data_file:
            demo_employees = list(csv.DictReader(test_data_file))
    except OSError as exc:
        st.error(f"Could not load employment verification test data: {exc}")
    else:
        if not demo_employees:
            st.info("No employment verification test records are available.")
        else:
            if is_admin:
                employee_by_label = {
                    f"{employee['employee_id']} — {employee['employee_name']}":
                        employee
                    for employee in demo_employees
                }
                with st.form("demo_employment_verification_form"):
                    selected_employee_label = st.selectbox(
                        "Select a fictional test employee",
                        tuple(employee_by_label),
                    )
                    demo_recipient = st.text_input(
                        "Recipient (optional)",
                        max_chars=160,
                    )
                    demo_purpose = st.text_input(
                        "Purpose (optional)",
                        max_chars=160,
                    )
                    generate_demo_letter = st.form_submit_button(
                        "Generate sample letter"
                    )
                employee = (
                    employee_by_label[selected_employee_label]
                    if generate_demo_letter
                    else None
                )
            else:
                user_employees = [
                    employee for employee in demo_employees
                    if employee.get("account_username", "").casefold()
                    == authenticated_user["username"].casefold()
                ]
                if len(user_employees) != 1:
                    st.info(
                        "There is no unique employee record linked to your "
                        "account. Contact an administrator to verify the "
                        "account-to-employee mapping."
                    )
                    employee = None
                    generate_demo_letter = False
                else:
                    employee = user_employees[0]
                    st.write(
                        f"Employee: **{employee['employee_name']}** "
                        f"({employee['employee_id']})"
                    )
                    with st.form("demo_employment_verification_form"):
                        demo_recipient = st.text_input(
                            "Recipient (optional)",
                            max_chars=160,
                        )
                        demo_purpose = st.text_input(
                            "Purpose (optional)",
                            max_chars=160,
                        )
                        generate_demo_letter = st.form_submit_button(
                            "Generate sample letter"
                        )

            if generate_demo_letter and employee is not None:
                start_date = date.fromisoformat(employee["start_date"])
                end_date = (
                    date.fromisoformat(employee["end_date"])
                    if employee["end_date"]
                    else None
                )
                employment_period = (
                    f"from {start_date.isoformat()} to {end_date.isoformat()}"
                    if end_date
                    else f"since {start_date.isoformat()}"
                )
                recipient_line = (
                    f"To: {demo_recipient.strip()}\n\n"
                    if demo_recipient.strip()
                    else ""
                )
                demo_letter = [
                    "SAMPLE — NOT VALID FOR EMPLOYMENT VERIFICATION",
                    "SmartHR Solutions",
                    "Employment Verification — TEST DATA",
                    f"Date: {date.today().isoformat()}",
                    "",
                    recipient_line.rstrip(),
                    (
                        "Dear Verification Recipient,"
                        if demo_recipient.strip()
                        else "To Whom It May Concern,"
                    ),
                    "",
                    (
                        f"This sample confirms that {employee['employee_name']} "
                        f"{'is currently employed' if not end_date else 'was employed'} "
                        f"by SmartHR Solutions as a {employee['job_title']} in the "
                        f"{employee['department']} department "
                        f"({employee['employment_type'].lower()}) {employment_period}."
                    ),
                    f"Sample employment status: {employee['employment_status']}.",
                    (
                        "Annual base salary: "
                        f"${int(employee['annual_salary_usd']):,} USD "
                        "(fictional test value; not for official use)."
                    ),
                ]
                if demo_purpose.strip():
                    demo_letter.append(f"Purpose: {demo_purpose.strip()}.")
                demo_letter.extend([
                    "",
                    "This document was generated from fictional test data. "
                    "It is not an official company statement and cannot be used "
                    "to establish employment, income, or identity.",
                    "",
                    "SAMPLE — NOT VALID FOR EMPLOYMENT VERIFICATION",
                ])
                st.session_state.demo_employment_verification_letter = "\n".join(
                    line for line in demo_letter if line is not None
                )
                st.session_state.demo_employment_verification_filename = (
                    f"sample_employment_verification_"
                    f"{employee['employee_id'].lower()}.txt"
                )

            if st.session_state.get("demo_employment_verification_letter"):
                st.text_area(
                    "Sample letter preview",
                    value=st.session_state.demo_employment_verification_letter,
                    height=360,
                    disabled=True,
                )
                st.download_button(
                    "Download sample letter",
                    data=st.session_state.demo_employment_verification_letter,
                    file_name=st.session_state.demo_employment_verification_filename,
                    mime="text/plain",
                )

elif active_page == "Manage Users" and authenticated_user["role"] == "admin":
    st.subheader("Manage Users")
    st.caption("Create accounts, reset passwords, and disable or re-enable access.")

    with st.form("create_user_form"):
        new_username = st.text_input("New username", max_chars=32)
        new_password = st.text_input(
            "Temporary password (minimum 12 characters)",
            type="password",
            max_chars=256,
        )
        new_role = st.selectbox("Role", ("user", "admin"))
        create_user_submitted = st.form_submit_button("Create account")

    if create_user_submitted:
        try:
            create_user(new_username, new_password, role=new_role)
            st.success(f"Account '{new_username.strip()}' created.")
        except ValueError as exc:
            st.error(str(exc))

    users = list_users()
    if users:
        st.dataframe(
            [
                {
                    "Username": user["username"],
                    "Role": user["role"],
                    "Active": user["active"],
                    "Created (UTC)": user["created_at"],
                }
                for user in users
            ],
            hide_index=True,
            use_container_width=True,
        )

        user_by_label = {
            f"{user['username']} ({user['role']}{', disabled' if not user['active'] else ''})":
                user
            for user in users
        }
        selected_label = st.selectbox(
            "Select account",
            tuple(user_by_label),
            key="managed_user_selection",
        )
        selected_user = user_by_label[selected_label]

        with st.form("reset_user_password_form"):
            reset_password_value = st.text_input(
                "New password (minimum 12 characters)",
                type="password",
                max_chars=256,
            )
            reset_submitted = st.form_submit_button("Reset password")
        if reset_submitted:
            try:
                reset_password(selected_user["id"], reset_password_value)
                st.success(f"Password reset for {selected_user['username']}.")
            except ValueError as exc:
                st.error(str(exc))

        if selected_user["id"] != authenticated_user["id"]:
            action = "Disable" if selected_user["active"] else "Re-enable"
            if st.button(f"{action} {selected_user['username']}"):
                try:
                    set_user_active(
                        selected_user["id"],
                        active=not selected_user["active"],
                        actor_id=authenticated_user["id"],
                    )
                    st.success(
                        f"Account '{selected_user['username']}' "
                        f"{'disabled' if selected_user['active'] else 're-enabled'}."
                    )
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
