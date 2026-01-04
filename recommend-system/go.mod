module recommend-system

go 1.22

require (
	github.com/gin-gonic/gin v1.9.1
	github.com/redis/go-redis/v9 v9.4.0
	github.com/jackc/pgx/v5 v5.5.2
	github.com/milvus-io/milvus-sdk-go/v2 v2.4.0
	github.com/spf13/viper v1.18.2
	go.uber.org/zap v1.26.0
	go.opentelemetry.io/otel v1.22.0
	go.opentelemetry.io/otel/trace v1.22.0
	go.opentelemetry.io/otel/exporters/jaeger v1.17.0
	google.golang.org/grpc v1.61.0
	google.golang.org/protobuf v1.32.0
	golang.org/x/time v0.5.0
	github.com/prometheus/client_golang v1.18.0
)

