# Make this project yours

This extension was built with Codex assistance. Run it, change it and be ready to
explain what you did before describing it as your own independent implementation.
It extends your original chatbot; it is not evidence of professional CRM experience.

## First session: 30–45 minutes

1. Run the fictional demo and inspect the four customer records and eight enquiries.
2. Submit a second question for Alex using the same email. Explain why it adds an
   enquiry but does not add a second customer.
3. Try `Can I cancel my order?` and an entirely new question. Explain why both are
   flagged for review in demo mode (conflicting labels versus no pattern).
4. Resolve an enquiry, then reopen it. Check how the open count changes.
5. Read `crm.capture`: identify the transaction, unique keys and parameterised SQL.

## Your first changes

- Add a `priority` field with three allowed values and a test for invalid input.
- Write a SQL query counting open enquiries by intent.
- Add three new paraphrases and expected labels in a separate evaluation file;
  do not add them to training before evaluating them.
- Clean overlapping intent labels only after deciding what each label means.
- Connect a HubSpot test account and verify one create plus one find-existing run.

## Interview questions

1. Why store customers separately from enquiries? What does an email key miss
   (shared addresses, changed addresses and identity verification)?
2. Why is a successful reply not the same as a resolved enquiry? What evidence
   would be needed to measure time saved or resolution rate?
3. What happens if HubSpot accepts a create but the network response is lost?
   Explain the lookup-before-create behaviour and remaining concurrency limits.

## Honest application wording, after you have worked through it

“I am extending my Python e-commerce chatbot into a personal CRM integration
project. The prototype records enquiries against customer contacts, supports
follow-up statuses and reports workload in SQL. It includes a contact-only HubSpot
adapter; [state whether you have actually verified it in a test account].”

Do not claim a live HubSpot integration, reduced handling time, production users,
or an independently built CRM until those statements are true. This project does
not retrospectively establish experience you had before starting it.

