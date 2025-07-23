# Код для scoring_engine.py (полный, правильный)
from typing import Dict, Any
class ScoringEngine:
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights; print(f"[Scoring Engine]: Инициализирован с весами: {self.weights}")
    def _score_technicals(self, tech_data: Dict) -> int:
        score = 0;
        if tech_data is None: return 0
        rsi, macd = tech_data.get('rsi'), tech_data.get('macd')
        if rsi is None or macd is None: return 0
        if rsi > 70: score -= 4
        if rsi < 30: score += 4
        if macd.get('crossover') == "BEARISH_CROSSOVER": score -= 6
        if macd.get('crossover') == "BULLISH_CROSSOVER": score += 6
        return score
    def _score_fundamentals(self, fund_data: Dict) -> int:
        if fund_data is None: return 0
        score, pe_ratio = 0, fund_data.get('pe_ratio')
        if pe_ratio is not None:
            if 0 < pe_ratio < 15: score += 5
            if pe_ratio > 40: score -= 5
        return score
    def calculate_final_score(self, full_analysis_data: Dict[str, Any]) -> int:
        tech_data = {'rsi': full_analysis_data.get('rsi'), 'macd': full_analysis_data.get('macd')}
        tech_score = self._score_technicals(tech_data)
        fund_score = self._score_fundamentals(full_analysis_data.get('fundamental'))
        final_score = (self.weights.get('technical', 0) * tech_score) + (self.weights.get('fundamental', 0) * fund_score)
        print(f"[Scoring Engine]: Тех. оценка={tech_score}, Фунд.={fund_score}. Итог (взвешенный): {final_score:.2f}")
        return int(round(final_score))