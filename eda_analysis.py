import pandas as pd
import numpy as np
import os
from zipfile import ZipFile
import warnings
warnings.filterwarnings('ignore')

# zip 파일에서 직접 데이터 읽기
zip_path = '/Users/corazzon/work/online-retail-rfm-eda/online_retail_ii.zip'

print("=" * 80)
print("Online Retail II Dataset - 기초 EDA")
print("=" * 80)

# zip 파일 내용 확인
with ZipFile(zip_path, 'r') as zip_file:
    file_list = zip_file.namelist()
    print(f"\nZip 파일 내용: {file_list}")
    
    # Excel 파일 읽기
    excel_file = [f for f in file_list if f.endswith('.xlsx')][0]
    print(f"\n데이터 파일: {excel_file}")
    
    with zip_file.open(excel_file) as file:
        # 엑셀 파일의 모든 시트 확인
        xl_file = pd.ExcelFile(file)
        print(f"\n시트 목록: {xl_file.sheet_names}")
        
        # 각 시트 읽기
        dfs = {}
        for sheet_name in xl_file.sheet_names:
            dfs[sheet_name] = pd.read_excel(xl_file, sheet_name=sheet_name)
            print(f"\n{'=' * 80}")
            print(f"시트: {sheet_name}")
            print(f"{'=' * 80}")
            
            df = dfs[sheet_name]
            
            # 기본 정보
            print(f"\n[1] 데이터 기본 정보")
            print(f"   - 행 개수: {len(df):,}")
            print(f"   - 열 개수: {len(df.columns)}")
            print(f"   - 컬럼: {list(df.columns)}")
            
            # 데이터 타입
            print(f"\n[2] 데이터 타입")
            print(df.dtypes)
            
            # 결측치 확인
            print(f"\n[3] 결측치 현황")
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            missing_df = pd.DataFrame({
                '결측치 개수': missing,
                '결측치 비율(%)': missing_pct
            })
            print(missing_df[missing_df['결측치 개수'] > 0])
            
            # 기초 통계량
            print(f"\n[4] 수치형 변수 기초 통계량")
            print(df.describe())
            
            # 상위 5개 행
            print(f"\n[5] 상위 5개 데이터")
            print(df.head())
            
            # 고유값 개수 (범주형 변수)
            print(f"\n[6] 각 컬럼별 고유값 개수")
            for col in df.columns:
                unique_count = df[col].nunique()
                print(f"   - {col}: {unique_count:,}")
            
            # Invoice 관련 분석
            if 'Invoice' in df.columns:
                print(f"\n[7] Invoice 분석")
                print(f"   - 총 거래 건수: {df['Invoice'].nunique():,}")
                canceled = df[df['Invoice'].astype(str).str.startswith('C', na=False)]
                print(f"   - 취소 거래 건수: {len(canceled):,} ({len(canceled)/len(df)*100:.2f}%)")
            
            # Customer 관련 분석
            if 'Customer ID' in df.columns:
                print(f"\n[8] Customer 분석")
                print(f"   - 총 고객 수: {df['Customer ID'].nunique():,}")
                print(f"   - 고객 정보 없음: {df['Customer ID'].isnull().sum():,}")
            
            # Country 관련 분석
            if 'Country' in df.columns:
                print(f"\n[9] Country 분석 (상위 10개)")
                country_counts = df['Country'].value_counts().head(10)
                for country, count in country_counts.items():
                    print(f"   - {country}: {count:,}")
            
            # Quantity 관련 분석
            if 'Quantity' in df.columns:
                print(f"\n[10] Quantity 분석")
                print(f"   - 평균 수량: {df['Quantity'].mean():.2f}")
                print(f"   - 중간값: {df['Quantity'].median():.2f}")
                print(f"   - 최소값: {df['Quantity'].min()}")
                print(f"   - 최대값: {df['Quantity'].max()}")
                negative_qty = df[df['Quantity'] < 0]
                print(f"   - 음수 수량 (반품): {len(negative_qty):,} ({len(negative_qty)/len(df)*100:.2f}%)")
            
            # Price 관련 분석
            if 'Price' in df.columns:
                print(f"\n[11] Price 분석")
                print(f"   - 평균 가격: £{df['Price'].mean():.2f}")
                print(f"   - 중간값: £{df['Price'].median():.2f}")
                print(f"   - 최소값: £{df['Price'].min():.2f}")
                print(f"   - 최대값: £{df['Price'].max():.2f}")
                zero_price = df[df['Price'] == 0]
                print(f"   - 가격 0: {len(zero_price):,} ({len(zero_price)/len(df)*100:.2f}%)")
            
            # InvoiceDate 관련 분석
            if 'InvoiceDate' in df.columns:
                print(f"\n[12] InvoiceDate 분석")
                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
                print(f"   - 기간: {df['InvoiceDate'].min()} ~ {df['InvoiceDate'].max()}")
                print(f"   - 총 기간: {(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} 일")

# 모든 시트 데이터 합치기
print("\n" + "=" * 80)
print("데이터 통합 및 저장")
print("=" * 80)

# 모든 시트를 하나의 데이터프레임으로 합치기
all_data = pd.concat(dfs.values(), ignore_index=True)

print(f"\n[통합 데이터 정보]")
print(f"   - 총 행 개수: {len(all_data):,}")
print(f"   - 총 열 개수: {len(all_data.columns)}")
print(f"   - 기간: {all_data['InvoiceDate'].min()} ~ {all_data['InvoiceDate'].max()}")
print(f"   - 총 고객 수: {all_data['Customer ID'].nunique():,}")
print(f"   - 총 거래 건수: {all_data['Invoice'].nunique():,}")

# CSV 파일로 저장
csv_path = '/Users/corazzon/work/online-retail-rfm-eda/online_retail_combined.csv'
all_data.to_csv(csv_path, index=False, encoding='utf-8')
print(f"\n✓ CSV 파일 저장 완료: {csv_path}")
print(f"   - 파일 크기: {os.path.getsize(csv_path) / (1024**2):.2f} MB")

# Parquet 파일로 저장
parquet_path = '/Users/corazzon/work/online-retail-rfm-eda/online_retail_combined.parquet'
all_data.to_parquet(parquet_path, index=False, engine='pyarrow')
print(f"\n✓ Parquet 파일 저장 완료: {parquet_path}")
print(f"   - 파일 크기: {os.path.getsize(parquet_path) / (1024**2):.2f} MB")

# 파일 크기 비교
csv_size = os.path.getsize(csv_path) / (1024**2)
parquet_size = os.path.getsize(parquet_path) / (1024**2)
compression_ratio = (1 - parquet_size / csv_size) * 100
print(f"\n[파일 크기 비교]")
print(f"   - CSV: {csv_size:.2f} MB")
print(f"   - Parquet: {parquet_size:.2f} MB")
print(f"   - 압축률: {compression_ratio:.1f}% 감소")

print("\n" + "=" * 80)
print("EDA 완료!")
print("=" * 80)
