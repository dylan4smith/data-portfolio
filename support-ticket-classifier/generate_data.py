"""
Generate synthetic support ticket dataset for multi-class text classification.

Produces realistic-looking customer support tickets labeled by department
(Billing, Technical, Account, Shipping, Product). Each ticket includes
a subject line, body text, priority level, and department label.
"""

import csv
import random
from pathlib import Path

SEED = 42
NUM_TICKETS = 2000

# --- Templates by department ---------------------------------------------------

TEMPLATES: dict[str, list[dict[str, str]]] = {
    "Billing": [
        {"subject": "Incorrect charge on my account", "body": "I noticed a charge of ${amount} on my last statement that I don't recognize. My account number is {acct}. Please investigate and issue a refund if this is an error."},
        {"subject": "Request for invoice copy", "body": "Could you please send me a copy of my invoice for {month}? I need it for my expense report. Account {acct}."},
        {"subject": "Payment not reflected", "body": "I made a payment of ${amount} on {date} via {method} but it still shows as outstanding on my account {acct}. Can you confirm receipt?"},
        {"subject": "Unexpected price increase", "body": "My monthly bill went from ${amount} to ${amount2} without any prior notice. I'm on the {plan} plan. Please explain this increase or revert it."},
        {"subject": "Duplicate charge", "body": "I was charged twice for the same transaction on {date}. The duplicate charge is ${amount}. Please refund the extra charge to my {method}."},
        {"subject": "Need to update payment method", "body": "I'd like to update the credit card on file for account {acct}. My current card ending in {card_last4} has expired."},
        {"subject": "Billing cycle change request", "body": "Can I switch my billing cycle from monthly to annual? I'd like to take advantage of the yearly discount for the {plan} plan."},
        {"subject": "Promo code not applied", "body": "I entered promo code {promo} during checkout but the discount wasn't applied. I was charged the full ${amount}. Please adjust my bill."},
    ],
    "Technical": [
        {"subject": "Application crashes on startup", "body": "After the latest update, the desktop app crashes immediately on launch. I'm running {os} version {version}. Error code: {error_code}. I've tried reinstalling but the issue persists."},
        {"subject": "Cannot connect to API", "body": "Our integration is returning {error_code} errors when calling the /v2/data endpoint. This started around {time} today. API key is still valid. Please check server-side logs."},
        {"subject": "Slow dashboard loading", "body": "The analytics dashboard takes over {seconds} seconds to load. This happens in both {browser} and {browser2}. Other pages load normally. Our dataset has about {rows} rows."},
        {"subject": "Data export failing", "body": "When I try to export reports as CSV, the download starts but fails at about 60%. File size should be around {size}MB. Using {browser} on {os}."},
        {"subject": "SSO login not working", "body": "Our team cannot log in via SSO since {date}. We use {sso_provider} as our identity provider. Regular email/password login still works. Error message: '{error_msg}'."},
        {"subject": "Webhook delivery failures", "body": "We're not receiving webhooks at our endpoint since {date}. The webhook URL is correctly configured. Our server returns 200 OK but your dashboard shows delivery failures."},
        {"subject": "Mobile app sync issues", "body": "Changes made in the mobile app on {os_mobile} aren't syncing to the web version. I've checked my internet connection and it's stable. App version {version}."},
        {"subject": "Report generation timeout", "body": "Generating the monthly summary report times out after {seconds} seconds. The report covers {rows} records. This worked fine last month with similar data volume."},
    ],
    "Account": [
        {"subject": "Reset my password", "body": "I've been locked out of my account ({email}) after too many failed login attempts. The password reset email isn't arriving. Can you manually reset it?"},
        {"subject": "Upgrade plan request", "body": "I'd like to upgrade from the {plan} to the {plan2} plan. We currently have {num_users} users and need the additional features. What's the process?"},
        {"subject": "Account merger request", "body": "We have two separate accounts ({email} and {email2}) that need to be merged. The {email2} account was created by mistake. Please consolidate under {email}."},
        {"subject": "Cancel my subscription", "body": "I need to cancel my {plan} subscription effective {date}. Please confirm the cancellation and any remaining balance. Account holder: {name}."},
        {"subject": "Add team members", "body": "I need to add {num_users} new users to our {plan} plan. Their email addresses are attached. Please confirm if this affects our pricing."},
        {"subject": "Change account owner", "body": "We need to transfer account ownership from {email} to {email2}. {name} has left the company and {name2} will be the new admin."},
        {"subject": "Data deletion request", "body": "Per our company policy, I'm requesting complete deletion of all data associated with account {email}. Please confirm when this is done and provide a confirmation reference."},
        {"subject": "Permission issue for team member", "body": "User {email} should have admin-level access but can only view dashboards. I've checked the role settings and they show 'Admin'. Please investigate."},
    ],
    "Shipping": [
        {"subject": "Order not received", "body": "My order #{order_id} was supposed to arrive by {date} but I still haven't received it. Tracking number {tracking} shows it's been in transit for {days} days."},
        {"subject": "Wrong item shipped", "body": "I ordered {item} but received {item2} instead. Order #{order_id}. I need the correct item sent and a return label for the wrong one."},
        {"subject": "Damaged package", "body": "My order #{order_id} arrived with significant damage to the packaging and the {item} inside is broken. I've attached photos. Please send a replacement."},
        {"subject": "Change shipping address", "body": "I need to update the shipping address for order #{order_id} from {city} to {city2}. The order hasn't shipped yet. New address: {address}."},
        {"subject": "Request expedited shipping", "body": "Is it possible to upgrade order #{order_id} to expedited shipping? I need it by {date}. Happy to pay the difference."},
        {"subject": "International shipping inquiry", "body": "I'm trying to place an order for delivery to {country}. Your website says shipping is unavailable for my region. Do you have plans to expand international shipping?"},
        {"subject": "Return shipping label needed", "body": "I'd like to return order #{order_id} ({item}). It doesn't fit our requirements. Could you please email a prepaid return label to {email}?"},
        {"subject": "Partial order received", "body": "Order #{order_id} was supposed to contain {num_items} items but only {num_items_received} arrived. Missing: {item}. The packing slip shows all items were included."},
    ],
    "Product": [
        {"subject": "Feature request: bulk import", "body": "We'd love the ability to bulk import records via CSV upload. Currently we have to add entries one by one, which is impractical for our {rows}-record dataset."},
        {"subject": "How to set up automated reports", "body": "I'm trying to configure scheduled reports to be emailed to my team every {frequency}. I can't find this option in the {plan} plan. Is it available or do I need to upgrade?"},
        {"subject": "Comparison with competitor", "body": "I'm evaluating your product against {competitor}. Can you provide a comparison of features, especially around {feature} and {feature2}? We're a {company_size}-person company."},
        {"subject": "Documentation unclear", "body": "The docs for the {feature} API endpoint are missing examples for {use_case}. I've spent {hours} hours trying to figure out the correct request format. Can you provide a working example?"},
        {"subject": "Suggestion for UI improvement", "body": "The current layout of the {feature} page requires too many clicks to accomplish basic tasks. Suggest adding a quick-action toolbar or keyboard shortcuts for power users."},
        {"subject": "Integration with third-party tool", "body": "Do you support integration with {tool}? We use it heavily for {use_case} and having a native connector would save us hours of manual data transfer each {frequency}."},
        {"subject": "Beta feature feedback", "body": "I've been testing the new {feature} beta. Overall it's great, but I found {num_issues} issues: the filter doesn't apply to grouped views, and export drops the last column."},
        {"subject": "Onboarding guidance needed", "body": "We just signed up for the {plan} plan with {num_users} users. Could you recommend the best way to onboard our team? We primarily need {feature} and {feature2}."},
    ],
}

# --- Fill-in values ------------------------------------------------------------

FILL_VALUES: dict[str, list[str]] = {
    "amount": ["29.99", "49.00", "99.95", "149.00", "199.50", "75.00", "12.99", "349.00"],
    "amount2": ["59.99", "79.00", "129.95", "199.00", "249.50"],
    "acct": ["AC-" + str(n) for n in range(100000, 100050)],
    "month": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
    "date": ["March 10", "March 15", "February 28", "last Monday", "two days ago", "yesterday", "March 1", "last Friday"],
    "method": ["credit card", "bank transfer", "PayPal", "ACH", "wire transfer"],
    "plan": ["Starter", "Professional", "Business", "Enterprise", "Growth"],
    "plan2": ["Professional", "Business", "Enterprise", "Growth", "Premium"],
    "card_last4": ["4242", "1234", "5678", "9012", "3456"],
    "promo": ["SAVE20", "WELCOME15", "SPRING25", "LOYALTY10", "ANNUAL30"],
    "os": ["Windows 11", "macOS Sonoma", "Ubuntu 22.04", "Windows 10", "macOS Ventura"],
    "os_mobile": ["iOS 17", "Android 14", "iOS 16", "Android 13"],
    "version": ["3.2.1", "4.0.0", "2.8.5", "3.1.0", "4.1.2"],
    "error_code": ["ERR_502", "TIMEOUT_408", "AUTH_401", "NULL_REF_500", "RATE_LIMIT_429"],
    "error_msg": ["Identity provider returned invalid response", "SAML assertion expired", "Certificate mismatch", "User not found in directory"],
    "time": ["9:30 AM EST", "2:15 PM UTC", "11:00 AM PST", "4:45 PM GMT"],
    "seconds": ["30", "45", "60", "90", "120"],
    "browser": ["Chrome", "Firefox", "Safari", "Edge"],
    "browser2": ["Firefox", "Safari", "Chrome", "Edge"],
    "rows": ["50,000", "100,000", "250,000", "500,000", "1,000,000"],
    "size": ["15", "50", "120", "250"],
    "sso_provider": ["Okta", "Azure AD", "Google Workspace", "OneLogin"],
    "email": ["jdoe@acmecorp.com", "admin@techfirm.io", "maria.g@retailco.com", "ops@startup.dev"],
    "email2": ["backup@acmecorp.com", "jsmith@techfirm.io", "team@retailco.com", "cto@startup.dev"],
    "name": ["John Doe", "Sarah Chen", "Maria Garcia", "Alex Kim"],
    "name2": ["Lisa Wang", "James Park", "Rachel Adams", "Miguel Torres"],
    "num_users": ["5", "10", "15", "25", "50"],
    "order_id": [str(n) for n in range(700100, 700150)],
    "tracking": ["1Z" + str(n) for n in range(9999900, 9999950)],
    "days": ["5", "7", "10", "14"],
    "item": ["wireless keyboard", "USB-C hub", "monitor stand", "laptop sleeve", "docking station"],
    "item2": ["mouse pad", "HDMI cable", "phone charger", "webcam", "headset"],
    "city": ["New York", "San Francisco", "Chicago", "Austin", "Seattle"],
    "city2": ["Los Angeles", "Denver", "Boston", "Portland", "Miami"],
    "address": ["123 Main St, Suite 400", "456 Oak Ave, Floor 2", "789 Pine Rd, Unit B"],
    "country": ["Brazil", "Japan", "Germany", "Australia", "India"],
    "num_items": ["3", "4", "5", "6"],
    "num_items_received": ["1", "2", "3", "4"],
    "frequency": ["Monday morning", "weekly", "bi-weekly", "monthly", "daily"],
    "competitor": ["Zendesk", "Freshdesk", "Salesforce", "HubSpot"],
    "feature": ["custom dashboards", "workflow automation", "advanced analytics", "team collaboration"],
    "feature2": ["role-based access", "API rate limits", "white-label options", "audit logging"],
    "company_size": ["25", "50", "150", "500"],
    "hours": ["3", "5", "8"],
    "tool": ["Slack", "Jira", "Salesforce", "Zapier", "Google Sheets"],
    "use_case": ["lead tracking", "project management", "customer onboarding", "compliance reporting"],
    "num_issues": ["2", "3", "4"],
}

PRIORITIES = ["Low", "Medium", "High", "Urgent"]
PRIORITY_WEIGHTS = [0.20, 0.40, 0.25, 0.15]

# Ambiguous tickets that blend two departments (makes classification harder)
AMBIGUOUS_TEMPLATES: list[dict[str, str]] = [
    {"subject": "Charged for item I returned", "body": "I returned order #{order_id} on {date} but I was charged ${amount} again. The {item} was sent back via {method}. Need a refund.", "department": "Billing"},
    {"subject": "Can't access account after plan change", "body": "I upgraded to {plan2} yesterday and now I get {error_code} when I try to log in. My email is {email}. Was the upgrade processed correctly?", "department": "Account"},
    {"subject": "Wrong amount on subscription renewal", "body": "My {plan} plan renewed at ${amount2} instead of ${amount}. I didn't authorize any changes. Account {acct}. Also the dashboard shows the old plan features.", "department": "Billing"},
    {"subject": "Shipping delay affecting our project", "body": "Order #{order_id} with {item} was supposed to arrive {date}. Our team can't proceed without it. Can we get an updated ETA? Also considering canceling if delayed further.", "department": "Shipping"},
    {"subject": "Technical issue with billing portal", "body": "The billing page returns {error_code} on {browser}. I can't view or download invoices for account {acct}. This has been happening since {date}.", "department": "Technical"},
    {"subject": "Need to cancel and get refund", "body": "I want to cancel my {plan} subscription and get a prorated refund for the remaining {days} days. Account {email}. Please process ASAP.", "department": "Account"},
    {"subject": "Product feedback and bug report", "body": "The new {feature} is great but it crashes when exporting to CSV on {os}. Error: {error_code}. Also, it would be useful to add {feature2} support.", "department": "Technical"},
    {"subject": "Shipped to wrong address, need rebill", "body": "Order #{order_id} went to {city} instead of {city2}. I need it reshipped and the extra charge on my {method} reversed. Account {acct}.", "department": "Shipping"},
]

# Generic / vague tickets that are harder to classify
GENERIC_TEMPLATES: list[dict[str, str]] = [
    {"subject": "Need help", "body": "I have an issue with my account {acct}. Something isn't working right. Can someone look into it?", "department": "Account"},
    {"subject": "Urgent issue", "body": "Please call me back regarding my recent order and billing. Account {email}. This needs immediate attention.", "department": "Billing"},
    {"subject": "Following up", "body": "I contacted you {date} about a problem but haven't heard back. My ticket was about {item} and charges on my account. Reference {acct}.", "department": "Billing"},
    {"subject": "Unhappy customer", "body": "I've been having nothing but problems. The {feature} doesn't work, I was overcharged ${amount}, and my last order #{order_id} never arrived. Please fix all of this.", "department": "Product"},
    {"subject": "Question about my account", "body": "Hi, I have some questions about my {plan} plan. Can you explain the features and also why my bill is ${amount}? Account {email}.", "department": "Account"},
]

LABEL_NOISE_RATE = 0.03  # 3% of tickets get a random wrong label


def fill_template(text: str, rng: random.Random) -> str:
    """Replace {placeholder} tokens with random values."""
    import re
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key in FILL_VALUES:
            return rng.choice(FILL_VALUES[key])
        return match.group(0)
    return re.sub(r"\{(\w+)\}", _replace, text)


def generate_tickets(n: int, seed: int = SEED) -> list[dict[str, str]]:
    """Generate *n* synthetic support tickets with balanced department labels.

    Includes ~15% ambiguous/generic tickets and 3% label noise to simulate
    real-world data quality, producing realistic (non-perfect) model accuracy.
    """
    rng = random.Random(seed)
    departments = list(TEMPLATES.keys())
    tickets: list[dict[str, str]] = []

    # Reserve slots for ambiguous and generic tickets (~15% of total)
    n_ambiguous = int(n * 0.10)
    n_generic = int(n * 0.05)
    n_standard = n - n_ambiguous - n_generic

    # Standard tickets from department-specific templates
    for i in range(n_standard):
        dept = departments[i % len(departments)]
        template = rng.choice(TEMPLATES[dept])
        subject = fill_template(template["subject"], rng)
        body = fill_template(template["body"], rng)
        priority = rng.choices(PRIORITIES, weights=PRIORITY_WEIGHTS, k=1)[0]
        tickets.append({
            "ticket_id": f"TKT-{10001 + i}",
            "subject": subject,
            "body": body,
            "priority": priority,
            "department": dept,
        })

    # Ambiguous cross-department tickets
    for i in range(n_ambiguous):
        template = rng.choice(AMBIGUOUS_TEMPLATES)
        subject = fill_template(template["subject"], rng)
        body = fill_template(template["body"], rng)
        priority = rng.choices(PRIORITIES, weights=PRIORITY_WEIGHTS, k=1)[0]
        tickets.append({
            "ticket_id": f"TKT-{10001 + n_standard + i}",
            "subject": subject,
            "body": body,
            "priority": priority,
            "department": template["department"],
        })

    # Generic / vague tickets
    for i in range(n_generic):
        template = rng.choice(GENERIC_TEMPLATES)
        subject = fill_template(template["subject"], rng)
        body = fill_template(template["body"], rng)
        priority = rng.choices(PRIORITIES, weights=PRIORITY_WEIGHTS, k=1)[0]
        tickets.append({
            "ticket_id": f"TKT-{10001 + n_standard + n_ambiguous + i}",
            "subject": subject,
            "body": body,
            "priority": priority,
            "department": template["department"],
        })

    # Apply label noise — randomly flip ~3% of labels
    for ticket in tickets:
        if rng.random() < LABEL_NOISE_RATE:
            wrong_depts = [d for d in departments if d != ticket["department"]]
            ticket["department"] = rng.choice(wrong_depts)

    rng.shuffle(tickets)
    return tickets


def main() -> None:
    out_path = Path(__file__).parent / "data" / "support_tickets.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tickets = generate_tickets(NUM_TICKETS)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticket_id", "subject", "body", "priority", "department"])
        writer.writeheader()
        writer.writerows(tickets)

    print(f"Generated {len(tickets)} tickets → {out_path}")
    for dept in sorted(TEMPLATES):
        count = sum(1 for t in tickets if t["department"] == dept)
        print(f"  {dept}: {count}")


if __name__ == "__main__":
    main()
