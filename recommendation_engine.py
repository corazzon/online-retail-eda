"""
상품 추천 엔진 모듈
- 협업 필터링 (Collaborative Filtering)
- 연관 규칙 분석 (Association Rules / Apriori)
- 인기도 기반 추천 (Popularity-based)
- RFM 기반 개인화 추천
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from collections import defaultdict
import json
import os
import pickle
from datetime import datetime

DATA_PATH = os.path.join(os.path.dirname(__file__), 'online_retail_combined.csv')
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'model_cache')


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_and_preprocess():
    """데이터 로드 및 전처리"""
    print("[1/6] 데이터 로딩 중...")
    df = pd.read_csv(DATA_PATH, dtype={'Customer ID': str, 'StockCode': str})

    # 기본 전처리
    df = df.dropna(subset=['Customer ID', 'Description'])
    df = df[df['Quantity'] > 0]
    df = df[df['Price'] > 0]
    df['Customer ID'] = df['Customer ID'].str.replace('.0', '', regex=False)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['Price']

    # 특수 상품코드 제거 (배송비 등)
    special_codes = ['POST', 'DOT', 'M', 'BANK CHARGES', 'PADS', 'C2', 'CRUK', 'D']
    df = df[~df['StockCode'].isin(special_codes)]

    # Description 정리
    df['Description'] = df['Description'].str.strip()

    print(f"   전처리 완료: {len(df):,} 건, 고객 {df['Customer ID'].nunique():,}명, 상품 {df['StockCode'].nunique():,}종")
    return df


def build_product_info(df):
    """상품 정보 테이블 구축"""
    print("[2/6] 상품 정보 구축 중...")
    product_info = df.groupby('StockCode').agg(
        Description=('Description', 'first'),
        AvgPrice=('Price', 'mean'),
        TotalSold=('Quantity', 'sum'),
        TotalRevenue=('TotalPrice', 'sum'),
        UniqueCustomers=('Customer ID', 'nunique'),
        OrderCount=('Invoice', 'nunique'),
    ).reset_index()

    product_info['AvgPrice'] = product_info['AvgPrice'].round(2)
    product_info['TotalRevenue'] = product_info['TotalRevenue'].round(2)
    return product_info


def build_rfm(df):
    """RFM 분석"""
    print("[3/6] RFM 분석 중...")
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    # RFM 점수 (1-5)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    # 고객 세그먼트
    def segment(row):
        if row['RFM_Score'] >= 13:
            return 'Champions'
        elif row['RFM_Score'] >= 10:
            return 'Loyal Customers'
        elif row['R_Score'] >= 4 and row['F_Score'] <= 2:
            return 'New Customers'
        elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
            return 'At Risk'
        elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
            return 'Lost'
        else:
            return 'Regular'

    rfm['Segment'] = rfm.apply(segment, axis=1)
    rfm['Monetary'] = rfm['Monetary'].round(2)
    return rfm


def build_popularity_recommendations(df, product_info, top_n=50):
    """인기도 기반 추천"""
    print("[4/6] 인기도 기반 추천 구축 중...")
    # 최근 3개월 데이터 기반 트렌드
    recent_date = df['InvoiceDate'].max() - pd.Timedelta(days=90)
    recent_df = df[df['InvoiceDate'] >= recent_date]

    trending = recent_df.groupby('StockCode').agg(
        RecentSales=('Quantity', 'sum'),
        RecentOrders=('Invoice', 'nunique'),
        RecentCustomers=('Customer ID', 'nunique'),
    ).reset_index()

    trending = trending.merge(product_info[['StockCode', 'Description', 'AvgPrice', 'TotalSold']], on='StockCode')

    # 인기도 점수: 판매량 + 주문수 + 고객수 종합
    trending['PopularityScore'] = (
        trending['RecentSales'].rank(pct=True) * 0.3 +
        trending['RecentOrders'].rank(pct=True) * 0.3 +
        trending['RecentCustomers'].rank(pct=True) * 0.4
    ).round(4)

    trending = trending.sort_values('PopularityScore', ascending=False).head(top_n)
    return trending


def build_collaborative_filtering(df, min_purchases=5):
    """아이템 기반 협업 필터링"""
    print("[5/6] 협업 필터링 모델 구축 중...")

    # 자주 구매되는 상품만 필터 (속도 최적화)
    item_counts = df.groupby('StockCode')['Customer ID'].nunique()
    popular_items = item_counts[item_counts >= min_purchases].index
    df_filtered = df[df['StockCode'].isin(popular_items)]

    # 고객-상품 매트릭스 (이진)
    user_item = df_filtered.groupby(['Customer ID', 'StockCode'])['Quantity'].sum().reset_index()
    user_item['Purchased'] = 1

    # 피벗 테이블
    pivot = user_item.pivot_table(
        index='Customer ID', columns='StockCode', values='Purchased', fill_value=0
    )

    # 아이템 간 코사인 유사도
    item_similarity = cosine_similarity(pivot.T)
    item_sim_df = pd.DataFrame(item_similarity, index=pivot.columns, columns=pivot.columns)

    return item_sim_df, pivot


def build_association_rules(df, min_support=0.02, min_confidence=0.1):
    """연관 규칙 분석 (Apriori)"""
    print("[6/6] 연관 규칙 분석 중...")
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    # 인보이스별 상품 리스트 (상위 500개 상품으로 제한)
    top_items = df.groupby('StockCode')['Invoice'].nunique().nlargest(500).index
    df_top = df[df['StockCode'].isin(top_items)]

    basket = df_top.groupby('Invoice')['StockCode'].apply(list).reset_index()
    # 최소 2개 이상 상품 포함 거래만
    basket = basket[basket['StockCode'].apply(len) >= 2]

    # 샘플링 (성능)
    if len(basket) > 10000:
        basket = basket.sample(10000, random_state=42)

    transactions = basket['StockCode'].tolist()

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)

    frequent_items = apriori(te_df, min_support=min_support, use_colnames=True)

    if len(frequent_items) == 0:
        print("   연관 규칙: 빈발 항목 없음, min_support 조정")
        frequent_items = apriori(te_df, min_support=0.01, use_colnames=True)

    if len(frequent_items) > 0:
        rules = association_rules(frequent_items, metric='confidence', min_threshold=min_confidence)
        rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
        rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
        rules = rules.sort_values('lift', ascending=False)
        print(f"   연관 규칙 {len(rules)}개 발견")
        return rules
    else:
        print("   연관 규칙을 찾을 수 없습니다.")
        return pd.DataFrame()


def build_content_based_filtering(product_info):
    """TF-IDF 기반 콘텐츠 기반 필터링"""
    print("[7/7] 콘텐츠 기반 필터링 구축 중...")
    tfidf = TfidfVectorizer(stop_words='english')
    # 설명을 벡터화
    tfidf_matrix = tfidf.fit_transform(product_info['Description'].fillna(''))
    # 코사인 유사도 계산
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    content_sim_df = pd.DataFrame(
        content_sim,
        index=product_info['StockCode'],
        columns=product_info['StockCode']
    )
    return content_sim_df


def get_cf_recommendations(stock_code, item_sim_df, product_info, top_n=10):
    """협업 필터링 기반 유사 상품 추천"""
    if stock_code not in item_sim_df.index:
        return []

    similar = item_sim_df[stock_code].sort_values(ascending=False).iloc[1:top_n + 1]
    results = []
    for code, score in similar.items():
        info = product_info[product_info['StockCode'] == code]
        if not info.empty:
            results.append({
                'StockCode': code,
                'Description': info.iloc[0]['Description'],
                'Price': float(info.iloc[0]['AvgPrice']),
                'Similarity': round(float(score), 4),
                'TotalSold': int(info.iloc[0]['TotalSold']),
                'Reason': '이 상품과 함께 구매된 패턴이 매우 유사합니다.'
            })
    return results


def get_association_recommendations(stock_code, rules_df, product_info, top_n=10):
    """연관 규칙 기반 추천 (함께 구매되는 상품)"""
    if rules_df.empty:
        return []

    matching = rules_df[rules_df['antecedents'].apply(lambda x: stock_code in x)]
    results = []
    seen = set()

    for _, row in matching.head(top_n * 2).iterrows():
        for code in row['consequents']:
            if code not in seen and code != stock_code:
                seen.add(code)
                info = product_info[product_info['StockCode'] == code]
                if not info.empty:
                    results.append({
                        'StockCode': code,
                        'Description': info.iloc[0]['Description'],
                        'Price': float(info.iloc[0]['AvgPrice']),
                        'Confidence': round(float(row['confidence']), 4),
                        'Lift': round(float(row['lift']), 2),
                        'TotalSold': int(info.iloc[0]['TotalSold']),
                        'Reason': f'동시 구매 확률: {(float(row["confidence"])*100):.1f}%'
                    })

    return results[:top_n]


def get_cb_recommendations(stock_code, content_sim_df, product_info, top_n=10):
    """콘텐츠 기반 유사 상품 추천 (TF-IDF)"""
    if stock_code not in content_sim_df.index:
        return []

    similar = content_sim_df[stock_code].sort_values(ascending=False).iloc[1:top_n + 1]
    results = []
    for code, score in similar.items():
        info = product_info[product_info['StockCode'] == code]
        if not info.empty:
            results.append({
                'StockCode': code,
                'Description': info.iloc[0]['Description'],
                'Price': float(info.iloc[0]['AvgPrice']),
                'Similarity': round(float(score), 4),
                'TotalSold': int(info.iloc[0]['TotalSold']),
                'Reason': '상품 설명(텍스트)이 매우 유사합니다.'
            })
    return results


def get_customer_recommendations(customer_id, df, item_sim_df, product_info, rfm, top_n=10):
    """고객 맞춤 추천 (구매 이력 + RFM 기반)"""
    customer_purchases = df[df['Customer ID'] == customer_id]['StockCode'].unique()

    if len(customer_purchases) == 0:
        return [], None

    # RFM 정보
    rfm_info = rfm[rfm['Customer ID'] == customer_id]
    rfm_data = rfm_info.iloc[0].to_dict() if not rfm_info.empty else None

    # 구매한 상품들의 유사 상품 점수 합산
    scores = defaultdict(float)
    for code in customer_purchases:
        if code in item_sim_df.index:
            similar = item_sim_df[code]
            for item, score in similar.items():
                if item not in customer_purchases:
                    scores[item] += score

    # 상위 추천
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for code, score in sorted_items:
        info = product_info[product_info['StockCode'] == code]
        if not info.empty:
            results.append({
                'StockCode': code,
                'Description': info.iloc[0]['Description'],
                'Price': float(info.iloc[0]['AvgPrice']),
                'Score': round(float(score), 4),
                'TotalSold': int(info.iloc[0]['TotalSold']),
            })

    return results, rfm_data


class RecommendationSystem:
    """추천 시스템 통합 클래스"""

    def __init__(self):
        self.df = None
        self.product_info = None
        self.rfm = None
        self.popularity = None
        self.item_sim_df = None
        self.user_item_pivot = None
        self.rules = None
        self.content_sim_df = None
        self.is_ready = False

    def initialize(self):
        """모델 초기화"""
        ensure_cache_dir()
        cache_file = os.path.join(CACHE_DIR, 'model_data.pkl')

        if os.path.exists(cache_file):
            print("캐시에서 모델 로딩 중...")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            self.df = cached['df']
            self.product_info = cached['product_info']
            self.rfm = cached['rfm']
            self.popularity = cached['popularity']
            self.item_sim_df = cached['item_sim_df']
            self.user_item_pivot = cached['user_item_pivot']
            self.rules = cached['rules']
            self.content_sim_df = cached.get('content_sim_df')
            self.is_ready = True
            print("모델 로딩 완료!")
            return

        self.df = load_and_preprocess()
        self.product_info = build_product_info(self.df)
        self.rfm = build_rfm(self.df)
        self.popularity = build_popularity_recommendations(self.df, self.product_info)
        self.item_sim_df, self.user_item_pivot = build_collaborative_filtering(self.df)
        self.rules = build_association_rules(self.df)
        self.content_sim_df = build_content_based_filtering(self.product_info)

        # 캐시 저장
        print("모델 캐시 저장 중...")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'df': self.df,
                'product_info': self.product_info,
                'rfm': self.rfm,
                'popularity': self.popularity,
                'item_sim_df': self.item_sim_df,
                'user_item_pivot': self.user_item_pivot,
                'rules': self.rules,
                'content_sim_df': self.content_sim_df,
            }, f)

        self.is_ready = True
        print("\n✅ 추천 시스템 초기화 완료!")

    def get_stats(self):
        """시스템 통계 (시각화 데이터 포함)"""
        # 국가별 매출 비중
        country_revenue = self.df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
        top_countries = {k: round(float(v), 2) for k, v in country_revenue.items()}

        # 월별 매출 추이
        monthly_revenue = self.df.resample('M', on='InvoiceDate')['TotalPrice'].sum()
        monthly_sales = {k.strftime('%Y-%m'): round(float(v), 2) for k, v in monthly_revenue.items()}

        return {
            'total_transactions': len(self.df),
            'total_customers': int(self.df['Customer ID'].nunique()),
            'total_products': int(self.df['StockCode'].nunique()),
            'date_range': f"{self.df['InvoiceDate'].min().strftime('%Y-%m-%d')} ~ {self.df['InvoiceDate'].max().strftime('%Y-%m-%d')}",
            'total_countries': int(self.df['Country'].nunique()),
            'total_revenue': round(float(self.df['TotalPrice'].sum()), 2),
            'association_rules_count': len(self.rules) if self.rules is not None else 0,
            'cf_products_count': len(self.item_sim_df) if self.item_sim_df is not None else 0,
            'cb_products_count': len(self.content_sim_df) if self.content_sim_df is not None else 0,
            'rfm_segments': self.rfm['Segment'].value_counts().to_dict() if self.rfm is not None else {},
            'top_countries': top_countries,
            'monthly_sales': monthly_sales,
        }

    def get_all_products(self, page=1, size=20, query=''):
        """전체 상품 목록 (페이지네이션 및 검색 지원)"""
        data = self.product_info.copy()
        
        if query:
            query = query.upper()
            mask = data['StockCode'].str.contains(query) | data['Description'].str.contains(query)
            data = data[mask]
        
        total_count = len(data)
        total_pages = (total_count + size - 1) // size
        
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        
        products = data.iloc[start_idx:end_idx].to_dict('records')
        
        return {
            'products': products,
            'total_count': total_count,
            'total_pages': total_pages,
            'current_page': page
        }

    def search_products(self, query, limit=20):
        """상품 검색"""
        query = query.upper()
        mask = (
            self.product_info['Description'].str.upper().str.contains(query, na=False) |
            self.product_info['StockCode'].str.upper().str.contains(query, na=False)
        )
        results = self.product_info[mask].sort_values('TotalSold', ascending=False).head(limit)
        return results.to_dict('records')

    def search_customers(self, query, limit=20):
        """고객 검색"""
        mask = self.rfm['Customer ID'].str.contains(query, na=False)
        results = self.rfm[mask].head(limit)
        return results.to_dict('records')

    def recommend_by_product(self, stock_code, top_n=10):
        """상품 기반 추천 (협업 필터링 + 연관 규칙 + 콘텐츠 기반)"""
        cf_recs = get_cf_recommendations(stock_code, self.item_sim_df, self.product_info, top_n)
        ar_recs = get_association_recommendations(stock_code, self.rules, self.product_info, top_n)
        cb_recs = get_cb_recommendations(stock_code, self.content_sim_df, self.product_info, top_n)

        # 상품 정보
        info = self.product_info[self.product_info['StockCode'] == stock_code]
        product_data = info.iloc[0].to_dict() if not info.empty else None

        return {
            'product': product_data,
            'collaborative_filtering': cf_recs,
            'association_rules': ar_recs,
            'content_based': cb_recs,
            'principles': {
                'cf': '아이템 기반 협업 필터링: 수많은 고객의 구매 이력을 바탕으로, 현재 상품과 가장 비슷한 구매 패턴을 가진 상품들을 수학적(코사인 유사도)으로 계산하여 추천합니다.',
                'ar': '연관 규칙 분석: 장바구니 데이터를 분석하여 "A를 사면 B도 살 확률이 높다"라는 규칙을 발견합니다. 동시 구매 빈도와 상관관계를 바탕으로 함께 구매할 만한 상품을 제안합니다.',
                'cb': '콘텐츠 기반 필터링: 상품의 설명(Description)을 텍스트 분석(TF-IDF)하여, 현재 상품과 가장 품질이나 특징이 유사한 상품을 추천합니다.'
            }
        }

    def recommend_by_customer(self, customer_id, top_n=10):
        """고객 기반 맞춤 추천"""
        recs, rfm_data = get_customer_recommendations(
            customer_id, self.df, self.item_sim_df, self.product_info, self.rfm, top_n
        )

        # 고객 구매 이력
        history = self.df[self.df['Customer ID'] == customer_id]
        top_purchases = history.groupby('StockCode').agg(
            Description=('Description', 'first'),
            TotalQuantity=('Quantity', 'sum'),
            TotalSpent=('TotalPrice', 'sum'),
        ).sort_values('TotalSpent', ascending=False).head(10).reset_index()

        return {
            'customer_id': customer_id,
            'rfm': rfm_data,
            'recommendations': recs,
            'purchase_history': top_purchases.to_dict('records'),
        }

    def get_popular_products(self, top_n=20, sort_by='popularity'):
        """인기 상품 추천 (다양한 정렬 기준 지원)"""
        if sort_by == 'revenue':
            return self.product_info.sort_values('TotalRevenue', ascending=False).head(top_n).to_dict('records')
        elif sort_by == 'sales':
            return self.product_info.sort_values('TotalSold', ascending=False).head(top_n).to_dict('records')
        elif sort_by == 'customers':
            return self.product_info.sort_values('UniqueCustomers', ascending=False).head(top_n).to_dict('records')
        else: # popularity (가중치 방식)
            return self.popularity.head(top_n).to_dict('records')

    def get_segment_recommendations(self, segment):
        """세그먼트별 추천 전략"""
        strategies = {
            'Champions': {
                'strategy': '충성도 보상 및 크로스셀',
                'description': '최고 가치 고객. 독점 혜택 및 신상품 우선 추천',
                'discount': '10% VIP 할인',
            },
            'Loyal Customers': {
                'strategy': '업셀 및 구독 유도',
                'description': '꾸준한 고객. 상위 상품 추천 및 번들 제안',
                'discount': '7% 로열 할인',
            },
            'New Customers': {
                'strategy': '온보딩 및 첫 구매 유도',
                'description': '신규 고객. 베스트셀러 및 입문 상품 추천',
                'discount': '15% 웰컴 쿠폰',
            },
            'At Risk': {
                'strategy': '재활성화 캠페인',
                'description': '이탈 위험 고객. 과거 선호 상품 리마인드',
                'discount': '20% 컴백 할인',
            },
            'Lost': {
                'strategy': '윈백 캠페인',
                'description': '이탈 고객. 대폭 할인 및 신상품 알림',
                'discount': '25% 특별 할인',
            },
            'Regular': {
                'strategy': '참여도 향상',
                'description': '일반 고객. 맞춤 추천으로 관심 유도',
                'discount': '5% 일반 할인',
            },
        }

        segment_customers = self.rfm[self.rfm['Segment'] == segment]

        # 해당 세그먼트 고객들이 많이 구매한 상품
        segment_customer_ids = segment_customers['Customer ID'].tolist()
        segment_purchases = self.df[self.df['Customer ID'].isin(segment_customer_ids)]
        top_products = segment_purchases.groupby('StockCode').agg(
            Description=('Description', 'first'),
            TotalSold=('Quantity', 'sum'),
            UniqueCustomers=('Customer ID', 'nunique'),
            AvgPrice=('Price', 'mean'),
        ).sort_values('UniqueCustomers', ascending=False).head(10).reset_index()

        return {
            'segment': segment,
            'customer_count': len(segment_customers),
            'strategy': strategies.get(segment, {}),
            'top_products': top_products.to_dict('records'),
            'avg_rfm': {
                'Recency': round(float(segment_customers['Recency'].mean()), 1),
                'Frequency': round(float(segment_customers['Frequency'].mean()), 1),
                'Monetary': round(float(segment_customers['Monetary'].mean()), 2),
            },
            'global_avg_rfm': {
                'Recency': round(float(self.rfm['Recency'].mean()), 1),
                'Frequency': round(float(self.rfm['Frequency'].mean()), 1),
                'Monetary': round(float(self.rfm['Monetary'].mean()), 2),
            }
        }

    def get_arpu_analysis(self):
        """ARPU & ARPPU 월별 분석
        - ARPU: 전체 등록 고객 기준 1인당 평균 매출
        - ARPPU: 해당 월 결제 고객 기준 1인당 평균 매출
        """
        df = self.df.copy()
        df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
        total_unique_customers = df['Customer ID'].nunique()

        monthly = df.groupby('YearMonth').agg(
            revenue=('TotalPrice', 'sum'),
            customers=('Customer ID', 'nunique'),
            orders=('Invoice', 'nunique'),
        ).reset_index()

        monthly['arpu'] = monthly['revenue'] / total_unique_customers
        monthly['arppu'] = monthly['revenue'] / monthly['customers']
        monthly['aov'] = monthly['revenue'] / monthly['orders']
        monthly['paying_rate'] = monthly['customers'] / total_unique_customers * 100

        return {
            'months': monthly['YearMonth'].tolist(),
            'arpu': [round(v, 2) for v in monthly['arpu'].tolist()],
            'arppu': [round(v, 2) for v in monthly['arppu'].tolist()],
            'aov': [round(v, 2) for v in monthly['aov'].tolist()],
            'revenue': [round(v, 2) for v in monthly['revenue'].tolist()],
            'customers': monthly['customers'].tolist(),
            'orders': monthly['orders'].tolist(),
            'paying_rate': [round(v, 1) for v in monthly['paying_rate'].tolist()],
            'total_customers': total_unique_customers,
        }

    def get_retention_analysis(self):
        """코호트 리텐션 분석 (고객수 기반 + 매출액 기반)"""
        df = self.df.copy()
        df['OrderMonth'] = df['InvoiceDate'].dt.to_period('M')

        # 각 고객의 첫 구매월 (코호트)
        cohorts = df.groupby('Customer ID')['OrderMonth'].min().reset_index()
        cohorts.columns = ['Customer ID', 'CohortMonth']
        df = df.merge(cohorts, on='Customer ID')

        # 코호트 인덱스 (경과 개월 수)
        df['CohortIndex'] = (df['OrderMonth'] - df['CohortMonth']).apply(lambda x: x.n)

        # ===== 1) 고객수 기반 리텐션 =====
        cust_data = df.groupby(['CohortMonth', 'CohortIndex'])['Customer ID'].nunique().reset_index()
        cust_pivot = cust_data.pivot(index='CohortMonth', columns='CohortIndex', values='Customer ID').fillna(0)
        cust_sizes = cust_pivot[0]
        cust_retention = cust_pivot.divide(cust_sizes, axis=0) * 100
        cust_retention = cust_retention.tail(12)

        customer_result = {}
        for cohort in cust_retention.index:
            cohort_str = str(cohort)
            vals = cust_retention.loc[cohort].dropna().tolist()
            customer_result[cohort_str] = {
                'cohort_size': int(cust_sizes.loc[cohort]),
                'retention': [round(v, 1) for v in vals]
            }

        # ===== 2) 매출액 기반 리텐션 =====
        rev_data = df.groupby(['CohortMonth', 'CohortIndex'])['TotalPrice'].sum().reset_index()
        rev_pivot = rev_data.pivot(index='CohortMonth', columns='CohortIndex', values='TotalPrice').fillna(0)
        rev_base = rev_pivot[0]
        rev_retention = rev_pivot.divide(rev_base, axis=0) * 100
        rev_retention = rev_retention.tail(12)

        revenue_result = {}
        for cohort in rev_retention.index:
            cohort_str = str(cohort)
            vals = rev_retention.loc[cohort].dropna().tolist()
            revenue_result[cohort_str] = {
                'cohort_revenue': round(float(rev_base.loc[cohort]), 2),
                'retention': [round(v, 1) for v in vals]
            }

        return {
            'customer': customer_result,
            'revenue': revenue_result
        }
