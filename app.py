"""
상품 추천시스템 Flask 웹 서버
"""

from flask import Flask, render_template, jsonify, request
from recommendation_engine import RecommendationSystem
import os

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

rec_sys = RecommendationSystem()


@app.after_request
def add_no_cache_headers(response):
    """브라우저 캐시 방지"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stats')
def api_stats():
    """시스템 통계"""
    return jsonify(rec_sys.get_stats())


@app.route('/api/search/products')
def api_search_products():
    """상품 검색"""
    query = request.args.get('q', '')
    if not query or len(query) < 2:
        return jsonify([])
    return jsonify(rec_sys.search_products(query))


@app.route('/api/search/customers')
def api_search_customers():
    """고객 검색"""
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    return jsonify(rec_sys.search_customers(query))


@app.route('/api/recommend/product/<stock_code>')
def api_recommend_product(stock_code):
    """상품 기반 추천"""
    return jsonify(rec_sys.recommend_by_product(stock_code))


@app.route('/api/products')
def api_products():
    """전체 상품 목록 (페이지네이션)"""
    page = request.args.get('page', 1, type=int)
    size = request.args.get('size', 20, type=int)
    query = request.args.get('q', '')
    return jsonify(rec_sys.get_all_products(page, size, query))


@app.route('/api/recommend/customer/<customer_id>')
def api_recommend_customer(customer_id):
    """고객 기반 추천"""
    return jsonify(rec_sys.recommend_by_customer(customer_id))


@app.route('/api/popular')
def api_popular():
    """인기 상품 (정렬 기능 추가)"""
    top_n = request.args.get('n', 20, type=int)
    sort_by = request.args.get('sort', 'popularity')
    return jsonify(rec_sys.get_popular_products(top_n, sort_by))


@app.route('/api/segment/<segment>')
def api_segment(segment):
    """세그먼트별 추천"""
    return jsonify(rec_sys.get_segment_recommendations(segment))


@app.route('/api/segments')
def api_segments():
    """전체 세그먼트 목록"""
    segments = rec_sys.rfm['Segment'].value_counts().to_dict()
    return jsonify(segments)



@app.route('/api/arpu')
def api_arpu():
    """ARPU 분석"""
    return jsonify(rec_sys.get_arpu_analysis())


@app.route('/api/retention')
def api_retention():
    """리텐션 분석"""
    return jsonify(rec_sys.get_retention_analysis())


if __name__ == '__main__':
    print("=" * 60)
    print("  상품 추천시스템 초기화 중...")
    print("=" * 60)
    rec_sys.initialize()
    print("=" * 60)
    print("  서버 시작: http://localhost:5001")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5001, debug=True)
