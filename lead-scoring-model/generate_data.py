"""
Generate synthetic B2B lead data for the lead scoring model.

Simulates a realistic CRM export with features that correlate with conversion
probability in ways that mirror real-world sales dynamics.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_leads(n_leads: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic B2B lead records with realistic conversion patterns.

    Args:
        n_leads: Number of lead records to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with lead features and a binary 'converted' target.
    """
    rng = np.random.default_rng(seed)

    # --- Firmographic features ---
    company_sizes = ["1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"]
    company_size = rng.choice(company_sizes, size=n_leads, p=[0.15, 0.25, 0.25, 0.15, 0.10, 0.10])

    industries = [
        "Technology", "Financial Services", "Healthcare",
        "Manufacturing", "Retail", "Professional Services", "Education"
    ]
    industry = rng.choice(industries, size=n_leads, p=[0.20, 0.15, 0.12, 0.13, 0.15, 0.15, 0.10])

    # --- Engagement features ---
    website_visits = rng.poisson(lam=5, size=n_leads)
    pages_per_session = rng.uniform(1.0, 12.0, size=n_leads).round(1)
    email_opens = rng.poisson(lam=3, size=n_leads)
    email_clicks = np.minimum(rng.poisson(lam=1, size=n_leads), email_opens)
    content_downloads = rng.poisson(lam=1.5, size=n_leads)
    webinar_attended = rng.binomial(1, 0.25, size=n_leads)
    demo_requested = rng.binomial(1, 0.15, size=n_leads)

    # --- Lead source ---
    sources = ["Organic Search", "Paid Ads", "Referral", "Social Media", "Event", "Direct"]
    lead_source = rng.choice(sources, size=n_leads, p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.15])

    # --- Time-based ---
    days_since_first_touch = rng.integers(1, 180, size=n_leads)
    days_since_last_activity = rng.integers(0, 90, size=n_leads)

    # --- Build conversion probability (logistic-style) ---
    # Base score set negative to target ~28% overall conversion rate
    score = np.full(n_leads, -2.5)

    # Engagement signals increase conversion likelihood
    score += 0.10 * website_visits
    score += 0.05 * pages_per_session
    score += 0.15 * email_clicks
    score += 0.20 * content_downloads
    score += 0.60 * webinar_attended
    score += 1.20 * demo_requested

    # Company size effect
    size_map = {"1-10": -0.3, "11-50": 0.0, "51-200": 0.2, "201-500": 0.3, "501-1000": 0.4, "1000+": 0.5}
    score += np.array([size_map[s] for s in company_size])

    # Industry effect
    ind_map = {
        "Technology": 0.3, "Financial Services": 0.2, "Healthcare": 0.1,
        "Manufacturing": 0.0, "Retail": -0.1, "Professional Services": 0.15, "Education": -0.2
    }
    score += np.array([ind_map[i] for i in industry])

    # Source effect
    src_map = {
        "Organic Search": 0.1, "Paid Ads": -0.1, "Referral": 0.5,
        "Social Media": -0.2, "Event": 0.3, "Direct": 0.0
    }
    score += np.array([src_map[s] for s in lead_source])

    # Recency: recent activity is a positive signal
    score -= 0.01 * days_since_last_activity

    # Add noise
    score += rng.normal(0, 0.5, size=n_leads)

    # Convert to probability via sigmoid
    prob = 1 / (1 + np.exp(-score))
    converted = rng.binomial(1, prob)

    df = pd.DataFrame({
        "company_size": company_size,
        "industry": industry,
        "lead_source": lead_source,
        "website_visits": website_visits,
        "pages_per_session": pages_per_session,
        "email_opens": email_opens,
        "email_clicks": email_clicks,
        "content_downloads": content_downloads,
        "webinar_attended": webinar_attended,
        "demo_requested": demo_requested,
        "days_since_first_touch": days_since_first_touch,
        "days_since_last_activity": days_since_last_activity,
        "converted": converted,
    })

    return df


if __name__ == "__main__":
    output_path = Path(__file__).parent / "data" / "leads.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    leads = generate_leads()
    leads.to_csv(output_path, index=False)

    print(f"Generated {len(leads)} lead records -> {output_path}")
    print(f"Conversion rate: {leads['converted'].mean():.1%}")
    print(f"\nFeature summary:\n{leads.describe().round(2)}")
