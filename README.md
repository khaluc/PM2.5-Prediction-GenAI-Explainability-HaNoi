# Environment AI

Hệ thống giám sát chất lượng không khí Hà Nội theo giờ, dự báo PM2.5 bằng
Machine Learning, phát hiện bất thường, phân tích giả thuyết nguyên nhân, quản lý
cảnh báo và tạo báo cáo vận hành.

## Thành phần

- `data/raw/`: dữ liệu Open-Meteo/CAMS và thời tiết mới thu thập.
- `data/processed/`: dữ liệu sạch, feature, tập train/validation/test và kết quả phân tích.
- `notebooks/`: EDA, làm sạch, feature engineering và thử nghiệm mô hình.
- `src/collection/`: collector chất lượng không khí, thời tiết và giao thông.
- `src/preprocessing/`: làm sạch và tạo feature chuỗi thời gian.
- `src/models/`: huấn luyện, dự báo và phát hiện bất thường.
- `src/analysis/`: phân tích giả thuyết nguyên nhân dựa trên dữ liệu.
- `src/genai/`: giải thích dự báo ML bằng Groq GPT-OSS với guardrail và chế độ dự phòng.
- `src/alerts/`: tạo, chống trùng và xác nhận cảnh báo.
- `src/database/`: schema SQLAlchemy, repository PostgreSQL và writer idempotent.
- `src/services/`: repository dữ liệu, inference và báo cáo.
- `api/`: FastAPI backend.
- `dashboard/`: Flask dashboard.
- `artifacts/models/`: model đã huấn luyện.
- `tests/`: kiểm thử tự động.

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Biến môi trường cần thiết khi thu thập giao thông:

```dotenv
TOMTOM_API_KEY=...
```

Để bật Groq cho chức năng giải thích dự báo, thêm vào `.env`:

```dotenv
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-120b
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

Nếu chưa cấu hình key hoặc Groq trả dữ liệu không qua guardrail, API vẫn trả bản giải
thích xác định sẵn và ghi `generation.mode=deterministic_fallback`.

## Thu thập dữ liệu live

`config.yaml` cấu hình 8 điểm lấy mẫu đại diện tại Hoàn Kiếm, Ba Đình, Cầu Giấy,
Đống Đa, Hai Bà Trưng, Thanh Xuân, Long Biên và Hà Đông.

```powershell
python scripts/run_collection.py --past-hours 192 --forecast-hours 1
```

Collector lưu tối thiểu 192 giờ quan trắc quá khứ để tạo feature live. Dữ liệu tương
lai của nhà cung cấp không được lưu hoặc dùng cho dự báo trên dashboard.

Nguồn dữ liệu:

- Chất lượng không khí: Open-Meteo Air Quality API, miền `cams_global`.
- Thời tiết: Open-Meteo Forecast API.
- Giao thông tùy chọn: TomTom Traffic API.

CAMS là dữ liệu mô hình theo ô lưới, không phải cảm biến mặt đất. Trường `source`
được giữ xuyên suốt pipeline để thể hiện nguồn gốc này.

## Dữ liệu lịch sử và làm sạch

```powershell
python scripts/collect_hanoi_history.py --config config.yaml
python scripts/run_cleaning.py --config config.yaml
```

Pipeline xử lý timestamp, bản ghi trùng, giá trị âm, khoảng hợp lệ, đơn vị, thiếu dữ
liệu ngắn hạn và cờ tăng đột biến. Dữ liệu lịch sử sạch được lưu tại
`data/processed/air_quality_clean.csv`.

## Feature engineering

```powershell
python scripts/run_feature_engineering.py --config config.yaml
```

Feature chính gồm:

- PM2.5 lag 1, 2, 3, 6, 12, 24, 48, 72 và 168 giờ.
- Lag của PM10, CO, NO₂, SO₂, O₃ và thời tiết.
- Rolling mean/std/min/max đến 168 giờ.
- EWM, xu hướng, tỷ lệ PM2.5/PM10 và thành phần gió.
- Chu kỳ giờ, ngày trong tuần và tháng.
- Nhãn PM2.5 tại +1h, +3h và +6h.

Mọi lag và rolling chỉ dùng dữ liệu từ `t-1` trở về trước để tránh leakage.

## Chia dữ liệu và huấn luyện

```powershell
python scripts/run_time_split.py --config config.yaml
python scripts/run_training.py --config config.yaml
```

Dữ liệu được chia theo thời gian, không random. Pipeline so sánh Baseline, Random
Forest, XGBoost, LightGBM và LSTM bằng MAE, RMSE, R² và MAPE. Artifact hiện tại chọn
LightGBM và được lưu tại `artifacts/models/pm25_forecast.joblib`.

Kết quả test của model hiện tại:

| Horizon | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| +1h | 3.18 | 5.72 | 0.966 | 6.53% |
| +3h | 7.97 | 12.25 | 0.843 | 17.09% |
| +6h | 12.28 | 17.58 | 0.677 | 28.64% |

## Phân loại PM2.5

```powershell
python scripts/run_pollution_classification.py --config config.yaml
```

Đánh giá WHO 2021 dùng trung bình trượt 24 giờ với yêu cầu tối thiểu 18 giờ hợp lệ.
Dự báo theo giờ chỉ là mức sàng lọc, không được trình bày như đánh giá tuân thủ WHO.

## Phát hiện bất thường

```powershell
python scripts/run_anomaly_detection.py --config config.yaml
```

Hệ thống kết hợp luật phạm vi, bước nhảy, flatline, quan hệ PM2.5/PM10 và Isolation
Forest. Model được lưu tại `artifacts/models/anomaly_detector.joblib`.

## Phân tích giả thuyết nguyên nhân

```powershell
python scripts/run_cause_analysis.py --config config.yaml
```

Module so sánh sự kiện với baseline `station_id × month`, sau đó xếp hạng các điều
kiện có thể đóng góp như tích tụ diện rộng, ít khuếch tán, tín hiệu đốt cháy, aerosol
thứ cấp, bụi thô và điều kiện quang hóa. Kết quả là giả thuyết kèm evidence và giới
hạn dữ liệu, không phải kết luận nhân quả hay source apportionment.

## FastAPI

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

OpenAPI: `http://127.0.0.1:8000/docs`

| Method | Route | Chức năng |
|---|---|---|
| GET | `/stations` | Danh sách khu vực và PM2.5 mới nhất |
| GET | `/stations/{station_id}/latest` | Quan trắc mới nhất |
| GET | `/stations/{station_id}/history` | Lịch sử có phân trang |
| POST | `/predict` | Dự báo LightGBM +1h/+3h/+6h |
| POST | `/detect-anomaly` | Luật ngưỡng và Isolation Forest |
| POST | `/forecast-explanation` | Giải thích có kiểm soát cho dự báo +1h/+3h/+6h |
| GET | `/alerts` | Danh sách cảnh báo |
| POST | `/alerts/evaluate` | Đánh giá và tạo cảnh báo |
| POST | `/alerts/{alert_id}/acknowledge` | Xác nhận cảnh báo |
| POST | `/reports/generate` | Tạo báo cáo JSON, Markdown hoặc PDF |
| GET | `/reports/{report_id}/download` | Tải báo cáo PDF đã tạo |
| GET | `/news` | Tin môi trường có cache, lọc Tin mới/Trong nước/Quốc tế |
| GET | `/health/ready` | Kiểm tra dữ liệu và feature artifacts |
| GET | `/system/hourly-update` | Trạng thái cập nhật dữ liệu và dự báo tự động mỗi giờ |
| POST | `/system/hourly-update/run` | Yêu cầu chạy ngay một kỳ cập nhật |
| GET | `/system/database` | Backend lưu trữ, trạng thái import và số dòng từng bảng |

Ví dụ dự báo live:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType application/json `
  -Body '{"station_id":"HN_HA_DONG"}'
```

API dựng 113 feature từ tối thiểu 169 giờ quan trắc liên tục. Nếu chuỗi không đủ hoặc
thiếu đầu vào, API trả lỗi rõ ràng và không fallback sang feature cũ.

Ví dụ giải thích dự báo:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/forecast-explanation `
  -ContentType application/json `
  -Body '{"station_id":"HN_HA_DONG","horizon_hours":3,"use_llm":true}'
```

GenAI chỉ diễn đạt dữ kiện có trong kết quả ML, thời tiết hiện tại và phân tích giả
thuyết đã được hệ thống hỗ trợ. Các cụm như gió yếu, độ ẩm cao và ít mưa được trình
bày là điều kiện *có thể góp phần*, không phải kết luận nhân quả. PM2.5 theo giờ chỉ
là mức sàng lọc, không được gọi là AQI chính thức.

## Dashboard Flask

```powershell
python -m dashboard.app
```

Mở `http://127.0.0.1:8501`. Dashboard gồm:

- Hero PM2.5 live với thang màu động, SVG lá phổi/hạt bụi chuyển động, AQI và thời tiết hiện tại.
- Dự báo PM2.5 LightGBM tại +1h, +3h và +6h.
- Biểu đồ PM2.5 trong 24 giờ gần nhất.
- Bản đồ Leaflet toàn Hà Nội với 8 điểm lấy mẫu.
- Trạng thái bất thường và danh sách cảnh báo.
- Thẻ GenAI giải thích dự báo tại mốc +1h, +3h hoặc +6h.
- Nút tạo và tải báo cáo PDF 24 giờ, có biểu đồ PM2.5, khí tượng, chất lượng và nguồn dữ liệu.

### Cập nhật tự động mỗi giờ

Khi FastAPI khởi động, worker nền chạy một kỳ thu thập ngay lập tức, sau đó căn theo
giờ Hà Nội và chạy ở giây thứ 45 của mỗi đầu giờ. Mỗi kỳ cập nhật CAMS/Open-Meteo,
thời tiết và TomTom, ghi CSV theo cơ chế nguyên tử rồi chạy lại LightGBM cho toàn bộ
điểm lấy mẫu. Nếu nhà cung cấp lỗi, worker ghi trạng thái và thử lại sau 5 phút.

Dashboard kiểm tra trạng thái mỗi 30 giây. Khi `last_success_at` thay đổi, dữ liệu hiện
tại, biểu đồ, bất thường và dự báo +1h/+3h/+6h được tải lại tự động; không cần F5.
Các biến `HOURLY_UPDATE_*` trong `.env` điều khiển lịch, độ trễ và file trạng thái.
- Trang `/news` tổng hợp tiêu đề, ngày, ảnh và mô tả ngắn từ Môi Trường Thủ Đô, có liên kết về bài gốc.

Trang tin dùng cache cục bộ trong 30 phút (`NEWS_CACHE_TTL_SECONDS=1800`) để hạn chế
request tới website nguồn. Nếu nguồn tạm thời lỗi, API sử dụng bản lưu gần nhất và đánh dấu
`stale=true`; hệ thống không crawl hoặc lưu toàn văn bài viết.

Giải thích GenAI còn sử dụng dữ liệu TomTom Flow Segment gần từng điểm lấy mẫu:
`currentSpeed`, `freeFlowSpeed`, thời gian di chuyển, `confidence`, trạng thái đóng
đường và tỷ lệ ùn tắc suy ra. Chỉ bản ghi gần thời điểm quan trắc trong giới hạn
`TRAFFIC_MAX_AGE_MINUTES` mới được đưa vào context. Tín hiệu ùn tắc chỉ trở thành
điều kiện có thể góp phần khi `confidence >= 0.7`; đây không phải kết luận giao
thông gây ra PM2.5 và không đại diện toàn quận/thành phố.

## Kiểm thử

```powershell
python -m pytest -q
```

## PostgreSQL

Database dùng PostgreSQL 16, migration Alembic và cơ chế `ON CONFLICT DO UPDATE` để
scheduler chạy lại không tạo dữ liệu trùng. Timestamp được lưu bằng `TIMESTAMPTZ`.

Khởi động database và nhập dữ liệu CSV hiện có:

```powershell
docker compose up -d db
python -m alembic upgrade head
python scripts/init_database.py
```

Sau khi `initial_import.completed=true`, API tự ưu tiên PostgreSQL. Nếu database chưa sẵn
sàng hoặc import chưa xong, API tiếp tục đọc CSV để dashboard không bị gián đoạn. Collector
mỗi giờ ghi song song vào PostgreSQL và CSV; dự báo LightGBM cùng kết quả bất thường cũng
được lưu theo thời điểm phát hành và phiên bản mô hình.

Kiểm tra:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/system/database
```

## Docker

```powershell
docker compose up api dashboard
```

### AWS EC2 production

Bộ production dành cho AWS EC2 dùng Nginx, Gunicorn, FastAPI, PostgreSQL private network,
healthcheck và backup tự động:

```bash
cp .env.production.example .env.production
bash deploy/aws/deploy.sh
```

Xem hướng dẫn tại [`deploy/aws/README.md`](deploy/aws/README.md). Chỉ Nginx được public;
không mở các cổng `5432`, `8000` hoặc `8501` trong EC2 Security Group.
