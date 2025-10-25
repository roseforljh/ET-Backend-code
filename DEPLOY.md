# EzTalk Proxy Docker Compose 部署指南

## 快速开始

### 1. 环境准备

确保您的VPS已安装：
- Docker
- Docker Compose

```bash
# 安装 Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. 部署步骤

```bash
# 1. 上传项目到VPS
scp -r backdAiTalk/ user@your-vps-ip:/opt/

# 2. 登录VPS并进入项目目录
ssh user@your-vps-ip
cd /opt/backdAiTalk

# 3. 配置环境变量
cp .env.example .env
nano .env  # 编辑配置文件

# 4. 启动服务
docker-compose up -d

# 5. 查看日志
docker-compose logs -f
```

### 3. 必需配置

编辑 `.env` 文件，至少需要配置：

```bash
# 必需：Google AI Studio API 密钥
GOOGLE_API_KEY=your_actual_api_key_here

# 可选：如果需要网络搜索功能
GOOGLE_CSE_ID=your_google_cse_id

# 可选：如果需要 Google Cloud Storage
GCS_BUCKET_NAME=your_bucket_name
GCS_PROJECT_ID=your_project_id
GEMINI_ENABLE_GCS_UPLOAD=true
```

### 4. 常用命令

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看日志
docker-compose logs -f eztalk-proxy

# 查看服务状态
docker-compose ps

# 重新构建并启动
docker-compose up -d --build
```

### 5. 健康检查

服务启动后，可以通过以下方式验证：

```bash
# 检查服务状态
curl http://localhost:7860/health

# 查看API文档
# 浏览器访问: http://your-vps-ip:7860/docs
```

### 6. 防火墙配置

```bash
# Ubuntu/Debian
sudo ufw allow 7860

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=7860/tcp
sudo firewall-cmd --reload
```

### 7. 使用 Nginx 反向代理（可选）

创建 Nginx 配置文件 `/etc/nginx/sites-available/eztalk-proxy`：

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 支持 Server-Sent Events
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }
}
```

启用配置：
```bash
sudo ln -s /etc/nginx/sites-available/eztalk-proxy /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 8. SSL 证书配置（可选）

使用 Let's Encrypt 获取免费 SSL 证书：

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 9. 故障排除

```bash
# 查看容器状态
docker ps

# 查看详细日志
docker-compose logs eztalk-proxy

# 进入容器调试
docker-compose exec eztalk-proxy /bin/bash

# 检查端口占用
netstat -tlnp | grep 7860
```

### 10. 更新服务

```bash
# 拉取最新代码
git pull

# 重新构建并启动
docker-compose up -d --build

# 清理旧镜像
docker image prune -f
```

## 注意事项

1. **安全性**：确保 `.env` 文件权限设置为 600，避免泄露敏感信息
2. **备份**：定期备份配置文件和重要数据
3. **监控**：建议配置日志轮转和监控告警
4. **更新**：定期更新 Docker 镜像和依赖包

现在您可以使用 Docker Compose 轻松部署和管理 EzTalk Proxy 服务了！