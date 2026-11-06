"SmartHR Solutions, a 500-employee SaaS company in San Ramon, CA.”

These are the 4 policy categories and the sub topics under each category
1.IT Support:
Password reset process
VPN configuration
Software installation requests
MFA setup
Guest Wi-Fi access
Email configuration on mobile

2.HR Policies:
Leave policy (casual, sick, earned)
WFH guidelines
Reimbursement process
Code of conduct
Anti-harassment policy
Performance review cycle

3.Onboarding:
Day-one checklist
IT equipment allocation
Benefits enrollment
Buddy programme
Probation period

4.Payroll & Comp:
Pay-day schedule
Tax declaration process
Salary structure
Stock-option vesting


"You are an HR manager at SmartHR Solutions, a 500-employee SaaS
company. Write a detailed Leave Policy document covering: casual leave
(12 days/year), sick leave (8 days/year), earned leave (15 days/year),
maternity leave (26 weeks), paternity leave (2 weeks), and compensatory
off. Include eligibility, approval process, carry-forward rules, and
encashment rules. Write in a professional tone for an employee handbook."

## Run the SmartHR Support UI

From the `SmartDeskAI` directory, install the project requirements and start
the Streamlit app:

```bash
python -m pip install -r requirements.txt
streamlit run SmartDesk/smartHR_app.py
```

See the [SmartHR high-level design](./ARCHITECTURE.svg) and its editable
[Graphviz source](./ARCHITECTURE.dot) for the main application components,
data stores, and external integrations.
The [Hybrid RAG diagram](./HYBRID_RAG.svg) shows how the support chat combines
semantic and keyword retrieval to find grounded answers.

## User login

The app stores user accounts in `data/smarthr_auth.sqlite3`, using Argon2id
password hashes. On first startup, set `SMART_HR_ADMIN_USERNAME` and
`SMART_HR_ADMIN_PASSWORD` in the environment or local `.env` file to create the
initial admin account. Use a unique password with at least 12 characters.
These values are used only while the account database has no users; remove the
bootstrap password from the environment after the admin account is created.

Sign in with the initial admin account and use **Manage Users** in the sidebar
to create user/admin accounts, reset passwords, and disable or re-enable
accounts. Usernames are 3-32 characters. Passwords must be at least 12
characters. The first admin must be bootstrapped before the login screen can be
used. Keep the account database private and deploy the app only over HTTPS.

Set `OPENAI_API_KEY` in your environment or in the local `.env` file. On the
first support-chat request, the app loads every JSON file directly inside
`data/`, normalizes ticket records (`subject`/`description`) and FAQ records
(`question`/`answer`) into a common document shape, and builds one shared
vector and keyword index. It also reads JSONL chat data with `messages` arrays,
converting each row's last user and assistant messages to a question/answer
record. Add knowledge `.json` or `.jsonl` files there. Evaluation/configuration
JSON files outside `data/` are not indexed.
Never commit `.env` or enter passwords, MFA codes, or other sensitive
information in the chat. If the assistant cannot answer from the ticket
dataset, it can collect the employee's email, summary, description, category,
and priority and submit a Jira issue after the employee confirms. Configure
`JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, `JIRA_PROJECT_KEY`, and
`JIRA_ISSUE_TYPE` in the local `.env` file. `JIRA_PRIORITY_MAP` can optionally
map helpdesk priorities to names configured in your Jira project. The Jira
account must have permission to create issues in that project. Users can also
ask the assistant to list their Jira tickets by providing the same work email
used when submitting them; the configured Jira account must have permission to
view issues in the project. In **Support Chat**, users can check an individual
Jira issue by key (for example `HELP-123`) or list issues by requester email.
The app sidebar provides separate **Log Jira Issue** and **Query Jira Issues**
pages, **Company Directory**, **Holiday Calendar**, **Policy Documents**, and
**Support Chat**. All signed-in users have an **Employment Verification** test
page for generating sample letters using 100 fictional records in
`data/employment_verification_testdata.csv`, including fictional annual base
salaries. Regular users see no employee
selector; their username must match the record's `account_username` field
(for example, `employee0001`). If there is no unique match, the app does not
generate a letter. Admins can select any demo employee. These fictional sample
letters are prominently marked invalid and are not stored in the account
database. For official letters, admins can use **HR Employment Letter** with
HR-verified details; it does not use the fictional directory.
Policy Documents displays the Markdown files in
`data/policies/`. The holiday page reads the U.S. federal holidays and weekday
observances for 2026 and 2027 from `data/holiday_calendar.csv`; it is a
reference list, not a company-specific paid-holiday schedule. The directory uses
fictional sample employees from `data/company_directory.csv`, including reserved
fictional phone numbers and manager IDs displayed in its **Org Chart** tab.
Replace this file with approved company directory data before using it as an
actual employee directory. Never commit the `.env` file or share the API token.
