# Triển khai Environment AI trên AWS EC2

## Kiến trúc

```text
Internet -> EC2 Security Group 80/443 -> Nginx
                                           |
                                           +-- Flask dashboard :8501 (private)
                                                   |
                                                   +-- FastAPI :8000 (private)
                                                           |
                                                           +-- PostgreSQL :5432 (private)
```

Chỉ Nginx publish cổng `80`. Không mở public `5432`, `8000` hoặc `8501`.
API chỉ chạy một replica vì scheduler cập nhật theo giờ nằm trong lifespan của FastAPI.

## EC2 đề xuất

- Region: Singapore (`ap-southeast-1`).
- Amazon Linux 2023 x86_64.
- `m7i-flex.large` hoặc tối thiểu 2 vCPU, 4 GiB RAM.
- Root EBS `gp3` 30 GiB.
- Security Group: SSH 22 chỉ từ IP quản trị; HTTP 80 và HTTPS 443 từ Internet.

## Chạy lần đầu

```bash
cd /opt/environment-ai
cp .env.production.example .env.production
nano .env.production
bash deploy/aws/deploy.sh
```

Bắt buộc điền `POSTGRES_PASSWORD`, `TOMTOM_API_KEY` và `GROQ_API_KEY`.
Không commit hoặc đưa `.env.production` vào Docker image.

Kiểm tra:

```bash
curl -I http://127.0.0.1/
docker compose --env-file .env.production -f docker-compose.aws.yml ps
```

## Backup

```bash
bash deploy/aws/backup.sh
```

Cron hằng ngày lúc 02:15 UTC:

```cron
15 2 * * * cd /opt/environment-ai && bash deploy/aws/backup.sh >> artifacts/logs/backup.log 2>&1
```

Cài lịch cron theo cách idempotent (chạy lại không tạo dòng trùng):

```bash
bash deploy/aws/install_backup_cron.sh
```

Restore chỉ chạy sau khi đã tạo EBS snapshot và backup mới:

```bash
CONFIRM_RESTORE=yes bash deploy/aws/restore.sh backups/environment-TIMESTAMP.dump
```

## Tài liệu AWS

- Launch EC2: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-launch-instance-wizard.html
- Security Group: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/security-group-rules-reference.html
- Thay đổi EBS: https://docs.aws.amazon.com/ebs/latest/userguide/ebs-modify-volume.html
- AWS Budgets: https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html
