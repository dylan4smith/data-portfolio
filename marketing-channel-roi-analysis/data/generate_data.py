"""
Generate synthetic marketing campaign performance data.

Simulates 18 months of campaign data across multiple channels, regions,
and audience segments for a mid-size B2B SaaS company.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_CAMPAIGNS = 500

channels = ["Google Ads", "LinkedIn Ads", "Email", "Content/SEO", "Webinars", "Trade Shows"]
regions = ["Northeast", "Southeast", "Midwest", "West", "International"]
segments = ["Enterprise", "Mid-Market", "SMB"]
quarters = ["2024-Q3", "2024-Q4", "2025-Q1", "2025-Q2"]

# Channel-specific cost and conversion profiles
channel_profiles = {
    "Google Ads":    {"avg_spend": 4500, "spend_std": 2000, "base_ctr": 0.032, "base_cvr": 0.045, "avg_deal": 8000},
    "LinkedIn Ads":  {"avg_spend": 5500, "spend_std": 2500, "base_ctr": 0.018, "base_cvr": 0.065, "avg_deal": 15000},
    "Email":         {"avg_spend": 800,  "spend_std": 400,  "base_ctr": 0.045, "base_cvr": 0.080, "avg_deal": 10000},
    "Content/SEO":   {"avg_spend": 2000, "spend_std": 1000, "base_ctr": 0.025, "base_cvr": 0.035, "avg_deal": 9000},
    "Webinars":      {"avg_spend": 3500, "spend_std": 1500, "base_ctr": 0.060, "base_cvr": 0.090, "avg_deal": 12000},
    "Trade Shows":   {"avg_spend": 15000,"spend_std": 5000, "base_ctr": 0.010, "base_cvr": 0.120, "avg_deal": 25000},
}

segment_multipliers = {"Enterprise": 1.8, "Mid-Market": 1.0, "SMB": 0.5}

records = []
for i in range(N_CAMPAIGNS):
    channel = np.random.choice(channels, p=[0.25, 0.20, 0.20, 0.15, 0.12, 0.08])
    region = np.random.choice(regions)
    segment = np.random.choice(segments, p=[0.25, 0.45, 0.30])
    quarter = np.random.choice(quarters)

    profile = channel_profiles[channel]
    seg_mult = segment_multipliers[segment]

    spend = max(100, np.random.normal(profile["avg_spend"], profile["spend_std"]))
    impressions = int(spend * np.random.uniform(80, 200))
    ctr = profile["base_ctr"] * np.random.lognormal(0, 0.3)
    clicks = int(impressions * ctr)
    leads = int(clicks * np.random.uniform(0.05, 0.20))

    cvr = profile["base_cvr"] * np.random.lognormal(0, 0.25) * (seg_mult ** 0.3)
    conversions = int(leads * cvr)
    conversions = max(0, conversions)

    deal_value = profile["avg_deal"] * seg_mult * np.random.lognormal(0, 0.2)
    revenue = conversions * deal_value

    # Campaign duration in days
    duration = int(np.random.choice([7, 14, 21, 30, 45, 60], p=[0.10, 0.25, 0.20, 0.25, 0.12, 0.08]))

    records.append({
        "campaign_id": f"CMP-{i+1:04d}",
        "channel": channel,
        "region": region,
        "segment": segment,
        "quarter": quarter,
        "spend_usd": round(spend, 2),
        "impressions": impressions,
        "clicks": clicks,
        "leads_generated": leads,
        "conversions": conversions,
        "revenue_usd": round(revenue, 2),
        "duration_days": duration,
    })

df = pd.DataFrame(records)
df.to_csv("data/campaign_performance.csv", index=False)
print(f"Generated {len(df)} campaign records.")
print(f"Total spend: ${df['spend_usd'].sum():,.2f}")
print(f"Total revenue: ${df['revenue_usd'].sum():,.2f}")
print(f"\nChannel distribution:\n{df['channel'].value_counts()}")
