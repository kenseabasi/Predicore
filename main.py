import numpy as np
from fastapi import FastAPI, HTTPException
from scipy.stats import nbinom
from pydantic import BaseModel
from typing import Optional, Dict, List
from supabase import create_client, Client

# --- SETUP ---
app = FastAPI(title="PrediCore Engine API")

# Replace these with your actual Supabase credentials
SUPABASE_URL = "YOUR_SUPABASE_URL"
SUPABASE_KEY = "YOUR_SUPABASE_SERVICE_ROLE_KEY"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- MODELS ---
class MatchInput(BaseModel):
    match_id: str
    home_xg_rolling: float
    away_xg_rolling: float
    home_defensive_resilience: float
    away_defensive_resilience: float
    aggregate_deficit: Optional[float] = 0.0 # Goals needed to tie aggregate
    lineup_strength: float = 1.0 # 0.0 to 1.0

# --- ENGINE LOGIC ---
def calculate_nb_params(mu, alpha):
    """Converts mean(mu) and dispersion(alpha) to NegBinom n and p."""
    if mu <= 0: return 1, 0.99
    var = mu + alpha * mu**2
    p = mu / var
    n = mu**2 / (var - mu)
    return n, p

@app.post("/predict")
async def predict_match(data: MatchInput):
    try:
        # 1. Base Expected Goals (mu)
        # We adjust xG based on the opponent's defensive resilience
        home_mu = data.home_xg_rolling * (2.0 - data.away_defensive_resilience)
        away_mu = data.away_xg_rolling * (2.0 - data.home_defensive_resilience)
        
        # 2. Apply Aggregate Deficit Urgency (The 'U' Multiplier)
        # If a team is down, we inflate their expected shot density
        if data.aggregate_deficit > 0:
            urgency_mult = 1.0 + (data.aggregate_deficit * 0.18)
            home_mu *= urgency_mult
        
        total_mu = home_mu + away_mu
        
        # 3. Dispersion (Alpha) - Volatility Filter
        # Lower lineup strength increases chaos (alpha), flattening the probability
        alpha = 0.12 + (1.0 - data.lineup_strength) * 0.3
        
        # 4. Negative Binomial Calculation for Over 2.5
        n, p = calculate_nb_params(total_mu, alpha)
        # Prob(Over 2.5) = 1 - (PMF(0)+PMF(1)+PMF(2))
        prob_over_2_5 = 1 - (nbinom.pmf(0, n, p) + nbinom.pmf(1, n, p) + nbinom.pmf(2, n, p))
        
        # 5. Confidence Score
        # High alpha (volatility) reduces confidence
        confidence = max(0, min(1, 0.95 - (alpha * 1.5)))
        
        # 6. Prepare Response
        result = {
            "match_id": data.match_id,
            "market": "over_2.5",
            "model_p": round(float(prob_over_2_5), 3),
            "calibrated_p": round(float(prob_over_2_5 * 0.97), 3), # Minor bias adjustment
            "confidence": round(float(confidence), 3),
            "recommendation": "STAKE_HIGH" if confidence >= 0.74 else "PASS",
            "features": {
                "adjusted_mu": round(total_mu, 2),
                "alpha_volatility": round(alpha, 3)
            }
        }

        # 7. Push to Supabase
        supabase.table("predictions").insert(result).execute()

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
  
