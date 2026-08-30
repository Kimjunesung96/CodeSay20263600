import pandas as pd

class SurgePricingEngine:
    """
    수급 불균형 지표 정의 및 동적 인센티브(Surge Pricing) 산출 클래스
    """
    def __init__(self, base_fare: int = 4800, min_multiplier: float = 1.0, max_multiplier: float = 3.0):
        # 기본 요금 4800원, 최대 할증 3배로 설정
        self.base_fare = base_fare
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

    def calculate_imbalance(self, demand: float, supply: float) -> float:
        """수요 대비 공급 비율 (수급 불균형 지표) 산출"""
        if supply == 0:
            return float('inf') if demand > 0 else 1.0
        return demand / supply

    def get_multiplier(self, imbalance_ratio: float) -> float:
        """불균형 지표에 따른 할증 배수 계산"""
        # 공급이 충분하거나 수요가 적을 때 (비율 <= 1.0)
        if imbalance_ratio <= 1.0:
            return self.min_multiplier
        
        # 수요가 더 많을 경우 초과분에 비례하여 배수 증가
        multiplier = 1.0 + (imbalance_ratio - 1.0) * 0.4
        return min(max(multiplier, self.min_multiplier), self.max_multiplier)

    def apply_surge_pricing(self, df: pd.DataFrame) -> pd.DataFrame:
        print("동적 인센티브(Surge Pricing) 산출 중...")
        
        # 불균형 지표(imbalance_ratio) 및 할증 배수(surge_multiplier) 컬럼 생성
        df['imbalance_ratio'] = df.apply(
            lambda row: self.calculate_imbalance(row['predicted_demand'], row['available_taxis']), axis=1
        )
        df['surge_multiplier'] = df['imbalance_ratio'].apply(self.get_multiplier)
        
        # 최종 요금 산출
        df['final_fare'] = (self.base_fare * df['surge_multiplier']).astype(int)
        
        print("산출 완료!")
        return df

if __name__ == "__main__":
    # Module 3에서 넘어온 예측 수요와 현재 가용 택시를 가정한 샘플 데이터
    sample_data = pd.DataFrame({
        'h3_index': ['8830e1ca2bfffff'] * 3,
        'time_bucket': ['18:00', '18:05', '18:10'],
        'predicted_demand': [10, 50, 5],   # 18:05에 수요 폭증 가정
        'available_taxis': [12, 10, 20]    # 18:05에 택시 부족
    })
    
    engine = SurgePricingEngine()
    result_df = engine.apply_surge_pricing(sample_data)
    
    print("\n--- 동적 요금제 적용 결과 ---")
    print(result_df[['time_bucket', 'predicted_demand', 'available_taxis', 'imbalance_ratio', 'surge_multiplier', 'final_fare']])