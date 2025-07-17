### 启动服务

  * **首次启动或需要重新构建镜像时：**

    ```bash
    docker compose up --build -d
    ```

  * **如果镜像已构建，仅启动容器：**

    ```bash
    docker compose up -d
    ```

### 管理服务

  * **查看实时日志：**

    ```bash
    docker compose logs -f
    ```

  * **强制重新创建容器（例如，在修改代码或配置文件后）：**

    ```bash
    docker compose up -d --force-recreate
    ```

### 停止服务

  * **停止并移除容器、网络和数据卷：**
    ```bash
    docker compose down
    ```