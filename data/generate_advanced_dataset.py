import pandas as pd
import random

random.seed(42)

def generate_dataset():
    data = []

    # --- 1. Cross-Class Keywords (pricing, demo, bug/error, interested, support) ---
    cross_class = [
        # pricing
        ("pricing details please", "High Intent"),
        ("pricing seems too high not interested", "Low Intent"),
        ("need pricing but not urgent", "Medium Intent"),
        ("just checking pricing not buying yet", "Inquiry"),
        ("pricing on the website is completely wrong and confusing", "Complaint"),
        ("pricing is way beyond our budget right now", "Low Intent"),
        ("client asked for pricing during call", "High Intent"), # CRM note
        ("what is your pricing model?", "Inquiry"),
        ("send over the pricing tier list", "Medium Intent"),
        
        # demo
        ("need to schedule a demo immediately", "High Intent"),
        ("demo was okay, we will think about it", "Medium Intent"),
        ("can I get a recorded demo link?", "Inquiry"),
        ("your demo environment kept crashing", "Complaint"),
        ("not ready for a demo yet", "Low Intent"),
        ("demo looked cool but we don't have the budget", "Low Intent"),
        ("lead requested demo for next tuesday", "Medium Intent"), # CRM note
        ("how long is the typical demo?", "Inquiry"),
        
        # bug/error
        ("found a bug but I still want to buy", "High Intent"), # Mixed signal
        ("how do I report a non-critical bug?", "Inquiry"),
        ("this bug is unacceptable, cancelling my account", "Complaint"),
        ("getting an error 500 on checkout, fix it so I can pay", "High Intent"),
        ("not interested, heard your app has too many bugs", "Low Intent"),
        ("is there a known error with active directory sync?", "Inquiry"),
        ("client experienced error during pilot, evaluating", "Medium Intent"), # CRM note
        
        # interested
        ("highly interested, send contract", "High Intent"),
        ("interested but budget is an issue", "Medium Intent"),
        ("not interested at all", "Low Intent"),
        ("client mentioned interest but no budget currently", "Medium Intent"), # CRM note
        ("are other companies in our industry interested in this?", "Inquiry"),
        ("I was interested until support was rude to me", "Complaint"),
        
        # support
        ("need support to upgrade my plan to enterprise", "High Intent"),
        ("when is support available?", "Inquiry"),
        ("support ignored my ticket for 3 days", "Complaint"),
        ("support was helpful but we chose another tool", "Low Intent"),
        ("waiting on support resolution before making a decision", "Medium Intent")
    ]
    data.extend(cross_class)

    # --- 2. Ambiguous & Contradictory Samples (Mixed Signals) ---
    ambiguous = [
        ("system is slow but we still like it", "Medium Intent"),
        ("need demo but not urgent", "Medium Intent"),
        ("this looks good but maybe later", "Medium Intent"),
        ("pricing is high so probably not", "Low Intent"),
        ("love the features, hate the UI, will consider", "Medium Intent"),
        ("we need this yesterday but legal is blocking", "Medium Intent"),
        ("it's exactly what we need but we just signed with a competitor", "Low Intent"),
        ("terrible experience with sales, but the product works so we'll buy", "High Intent"),
        ("too expensive, unless you can do 50% off?", "Medium Intent"),
        ("I don't get how this works, but my boss wants me to buy it", "High Intent"),
        ("looks amazing, no budget till 2025", "Medium Intent"),
        ("not sure yet, need to think", "Medium Intent"),
        ("maybe later, not convinced", "Medium Intent"),
        ("let's talk next week", "Medium Intent"),
        ("checking options, not sure yet", "Medium Intent")
    ]
    data.extend(ambiguous)

    # --- 3. CRM-Style Internal Notes ---
    crm_notes = [
        ("client asked for pricing during call", "High Intent"),
        ("lead discussing internally with finance", "Medium Intent"),
        ("customer rejected proposal due to cost", "Low Intent"),
        ("user reported login issue during demo", "Complaint"),
        ("spoke with VP, they are evaluating 3 vendors including us", "Medium Intent"),
        ("POC successful, procurement is reviewing MSA", "High Intent"),
        ("champion left the company, deal on hold", "Low Intent"),
        ("inbound lead looking for basic integration docs", "Inquiry"),
        ("customer furious about ongoing downtime, churn risk", "Complaint"),
        ("budget approved for Q3, follow up in August", "Medium Intent"),
        ("wrong person, referred me to the CTO", "Low Intent"),
        ("lead unresponsive after 5 follow ups, closing out", "Low Intent"),
        ("verbal commitment secured on the call today", "High Intent"),
        ("client just wants to know if we have a mobile app", "Inquiry"),
        ("customer filed a feature request, not happy it's missing", "Complaint"),
        ("sent contract via docusign, waiting for signature", "High Intent"),
        ("met at conference, mild interest, add to nurture sequence", "Medium Intent"),
        ("lead wants a demo but not sure yet", "Medium Intent"),
        ("client said maybe later", "Medium Intent")
    ]
    data.extend(crm_notes)

    # --- 4. Noise Variations (typos, slang, short, long) ---
    noise = [
        ("prcing plz", "High Intent"),
        ("dmeo", "Inquiry"),
        ("idk maybe later bro", "Low Intent"),
        ("yo can u share pricing rn", "High Intent"),
        ("price?", "Inquiry"),
        ("help", "Inquiry"),
        ("broken", "Complaint"),
        ("buy", "High Intent"),
        ("nah", "Low Intent"),
        ("wanna test it out", "Medium Intent"),
        ("u guys have an api or what", "Inquiry"),
        ("this software is literally garbage ngl", "Complaint"),
        ("when u drop the new update hit me up", "Medium Intent"),
        ("pls send invoice rn", "High Intent"),
        ("app kinda buggy but ok for now", "Medium Intent"),
        ("need pricing and demo asap", "High Intent"),
        ("dashboard is crashing", "Complaint"),
        ("dashboard keeps crashing when I export", "Complaint"),
        ("need pricing and a demo immediately", "High Intent")
    ]
    data.extend(noise)

    # --- 5. Inquiry vs Complaint Distinctions ---
    distinctions = [
        ("how to fix login issue?", "Inquiry"),
        ("login not working again", "Complaint"),
        ("where is the billing portal?", "Inquiry"),
        ("my card was charged incorrectly", "Complaint"),
        ("can I reset my password?", "Inquiry"),
        ("password reset email never arrives, fix this", "Complaint"),
        ("what are the rate limits on the API?", "Inquiry"),
        ("API keeps aggressively rate limiting me, dropping requests", "Complaint")
    ]
    data.extend(distinctions)

    # --- 6. Hard Cases (Unclear, mixed sentiment, vague) ---
    hard_cases = [
        ("I saw your ad.", "Low Intent"),
        ("My boss told me to look at this.", "Inquiry"),
        ("We use Excel.", "Low Intent"),
        ("Is there a way to do the thing with the stuff?", "Inquiry"),
        ("Not sure why I'm getting these emails.", "Low Intent"),
        ("I guess it's fine.", "Medium Intent"),
        ("Whatever.", "Low Intent"),
        ("Can we talk?", "Inquiry"),
        ("Send it over.", "High Intent"), # Vague but actionable
        ("We are a 5 person team.", "Inquiry"), # Context missing
        ("Make it work.", "Complaint"),
        ("Do better.", "Complaint"),
        ("I'm leaving.", "Low Intent") 
    ]
    data.extend(hard_cases)

    # --- 7. Bulk Data to reach 600+ rows (Balanced) ---
    # We will procedurally generate realistic variations to avoid exact repetition
    
    bases_high = [
        "ready to proceed with the purchase", "send over the order form", "let's get started on onboarding",
        "we want to move forward with the enterprise plan", "urgent need to implement this week",
        "can we finalize the agreement today", "ready to sign off on the proposal", "process payment for the annual tier",
        "need licenses for my team immediately", "let's kick off the project", "we selected you as our vendor",
        "where do I wire the funds", "send the MSA for legal review ASAP"
    ]
    
    bases_medium = [
        "still in the evaluation phase", "need to discuss internally with stakeholders", "we are comparing you with a few others",
        "following up next quarter might be better", "waiting for budget approval", "checking with IT for security clearance",
        "looks decent but we aren't ready yet", "I need to present this to the board first", "putting this on pause for a month",
        "can you put me on an email cadence for updates?", "we're interested but timing is bad",
        "need some time to think it over", "let's touch base in a few weeks"
    ]
    
    bases_low = [
        "please remove me from your list", "we went with another solution", "we don't have the budget for this",
        "not a priority right now", "not interested at all", "we build this internally so no need",
        "too expensive for our small shop", "don't ever call this number again", "I am just a student doing a project",
        "we are too small to need this", "unsubscribe immediately please", "wrong person, try someone else",
        "we are happy with our current setup"
    ]
    
    bases_inquiry = [
        "do you offer 24/7 technical support", "how exactly does the machine learning work", "is there a limit to how many users we can add",
        "can we export data as a CSV or JSON", "what languages does the platform support", "does this integrate natively with Salesforce",
        "how long is the implementation timeline usually", "do you guys have a compliance page I can check",
        "can I set granular permissions for different roles", "do you provide a dedicated customer success manager",
        "what happens when we reach our API limit", "is it possible to white-label the dashboard"
    ]
    
    bases_complaint = [
        "the analytics dashboard is completely inaccurate", "I have been waiting on hold for hours", "your recent update deleted my saved filters",
        "mobile application crashes every time I open it", "extremely slow loading times in the EU region",
        "billing department messed up my invoice again", "the user interface is incredibly non-intuitive",
        "why am I getting error 403 on a public endpoint", "lost all my work because the auto-save failed",
        "your sales rep was extremely rude on the phone", "the documentation is severely out of date and useless",
        "I want a full refund for this month's downtime"
    ]

    prefixes = ["", "so ", "hello, ", "hi, ", "hey ", "quick question: ", "fyi: ", "note: ", "update: "]
    suffixes = ["", " please", " thanks", " asap", " rn", ".", "!", "..."]

    def augment(base_list, intent, target_count):
        aug_data = []
        for _ in range(target_count):
            base = random.choice(base_list)
            pref = random.choice(prefixes)
            suff = random.choice(suffixes)
            sentence = pref + base + suff
            aug_data.append((sentence.strip().lower(), intent))
        return aug_data

    # Current count approx 100 rows. We need ~500 more to hit 600-650.
    # ~100 per class spread.
    
    data.extend(augment(bases_high, "High Intent", 100))
    data.extend(augment(bases_medium, "Medium Intent", 100))
    data.extend(augment(bases_low, "Low Intent", 100))
    data.extend(augment(bases_inquiry, "Inquiry", 100))
    data.extend(augment(bases_complaint, "Complaint", 100))

    # Add a bit more CRM noise dynamically
    crm_templates = [
        ("lead called in, {}", "High Intent", bases_high),
        ("sync with client: {}", "Medium Intent", bases_medium),
        ("closed lost: {}", "Low Intent", bases_low),
        ("support ticket: {}", "Complaint", bases_complaint),
        ("pre-sales ques: {}", "Inquiry", bases_inquiry)
    ]
    
    for template, intent, pool in crm_templates:
        for _ in range(15):
            phrase = template.format(random.choice(pool))
            data.append((phrase.lower(), intent))

    # Deduplicate while preserving order (optional, but requested no duplicates)
    seen = set()
    unique_data = []
    for text, intent in data:
        if text not in seen:
            seen.add(text)
            unique_data.append((text, intent))

    # Output to CSV
    df = pd.DataFrame(unique_data, columns=['text', 'intent'])
    
    # Final check to ensure we hit the 600-650 target
    print(f"Generated {len(df)} unique rows.")
    print(df['intent'].value_counts())
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df.to_csv("c:/Users/ASUS/Desktop/IntentIQ/data/dataset.csv", index=False)
    print("dataset.csv updated successfully.")

if __name__ == "__main__":
    generate_dataset()
