package database

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"recommend-system/pkg/logger"
	"go.uber.org/zap"
)

// MilvusClient Milvus 向量数据库客户端
type MilvusClient struct {
	client     client.Client
	database   string
	collection string
}

// MilvusConfig Milvus 配置
type MilvusConfig struct {
	Address    string
	Port       int
	User       string
	Password   string
	Database   string
	Collection string
}

// NewMilvusClient 创建 Milvus 客户端
func NewMilvusClient(cfg *MilvusConfig) (*MilvusClient, error) {
	addr := fmt.Sprintf("%s:%d", cfg.Address, cfg.Port)

	ctx := context.Background()

	// 创建客户端
	c, err := client.NewClient(ctx, client.Config{
		Address:  addr,
		Username: cfg.User,
		Password: cfg.Password,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to milvus: %w", err)
	}

	logger.Info("Milvus connected",
		zap.String("address", addr),
		zap.String("database", cfg.Database),
	)

	return &MilvusClient{
		client:     c,
		database:   cfg.Database,
		collection: cfg.Collection,
	}, nil
}

// Close 关闭连接
func (m *MilvusClient) Close() error {
	if m.client != nil {
		logger.Info("Milvus connection closed")
		return m.client.Close()
	}
	return nil
}

// Client 返回原始客户端
func (m *MilvusClient) Client() client.Client {
	return m.client
}

// CreateCollection 创建集合
func (m *MilvusClient) CreateCollection(ctx context.Context, name string, dim int) error {
	// 定义 Schema
	schema := &entity.Schema{
		CollectionName: name,
		Description:    "Item embeddings for recommendation",
		Fields: []*entity.Field{
			{
				Name:       "item_id",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				AutoID:     false,
				TypeParams: map[string]string{
					"max_length": "64",
				},
			},
			{
				Name:     "embedding",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", dim),
				},
			},
			{
				Name:     "semantic_l1",
				DataType: entity.FieldTypeInt32,
			},
			{
				Name:     "semantic_l2",
				DataType: entity.FieldTypeInt32,
			},
			{
				Name:     "semantic_l3",
				DataType: entity.FieldTypeInt32,
			},
		},
	}

	err := m.client.CreateCollection(ctx, schema, 2) // 2 shards
	if err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	logger.Info("Milvus collection created", zap.String("name", name))
	return nil
}

// HasCollection 检查集合是否存在
func (m *MilvusClient) HasCollection(ctx context.Context, name string) (bool, error) {
	return m.client.HasCollection(ctx, name)
}

// DropCollection 删除集合
func (m *MilvusClient) DropCollection(ctx context.Context, name string) error {
	return m.client.DropCollection(ctx, name)
}

// CreateIndex 创建索引
func (m *MilvusClient) CreateIndex(ctx context.Context, collectionName, fieldName string) error {
	// 创建 HNSW 索引
	idx, err := entity.NewIndexHNSW(entity.IP, 16, 200)
	if err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	err = m.client.CreateIndex(ctx, collectionName, fieldName, idx, false)
	if err != nil {
		return fmt.Errorf("failed to create index on collection: %w", err)
	}

	logger.Info("Milvus index created",
		zap.String("collection", collectionName),
		zap.String("field", fieldName),
	)
	return nil
}

// LoadCollection 加载集合到内存
func (m *MilvusClient) LoadCollection(ctx context.Context, name string) error {
	return m.client.LoadCollection(ctx, name, false)
}

// ReleaseCollection 释放集合
func (m *MilvusClient) ReleaseCollection(ctx context.Context, name string) error {
	return m.client.ReleaseCollection(ctx, name)
}

// Insert 插入数据
func (m *MilvusClient) Insert(ctx context.Context, collectionName string, columns ...entity.Column) error {
	_, err := m.client.Insert(ctx, collectionName, "", columns...)
	return err
}

// Upsert 更新或插入数据
func (m *MilvusClient) Upsert(ctx context.Context, collectionName string, columns ...entity.Column) error {
	_, err := m.client.Upsert(ctx, collectionName, "", columns...)
	return err
}

// Search 向量搜索
func (m *MilvusClient) Search(
	ctx context.Context,
	collectionName string,
	vectors []entity.Vector,
	vectorField string,
	topK int,
	outputFields []string,
) ([]client.SearchResult, error) {
	// 搜索参数
	sp, _ := entity.NewIndexHNSWSearchParam(100) // ef

	results, err := m.client.Search(
		ctx,
		collectionName,
		nil,           // partitions
		"",            // expression
		outputFields,
		vectors,
		vectorField,
		entity.IP, // 内积
		topK,
		sp,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to search: %w", err)
	}

	return results, nil
}

// SearchByVector 通过向量搜索相似项
func (m *MilvusClient) SearchByVector(
	ctx context.Context,
	collectionName string,
	embedding []float32,
	topK int,
) ([]string, []float32, error) {
	vectors := []entity.Vector{entity.FloatVector(embedding)}

	results, err := m.Search(
		ctx,
		collectionName,
		vectors,
		"embedding",
		topK,
		[]string{"item_id"},
	)
	if err != nil {
		return nil, nil, err
	}

	if len(results) == 0 || results[0].ResultCount == 0 {
		return nil, nil, nil
	}

	result := results[0]
	itemIDs := make([]string, result.ResultCount)
	scores := make([]float32, result.ResultCount)

	for i := 0; i < result.ResultCount; i++ {
		// 获取 item_id
		itemIDCol, ok := result.Fields.GetColumn("item_id").(*entity.ColumnVarChar)
		if ok {
			itemIDs[i], _ = itemIDCol.ValueByIdx(i)
		}
		scores[i] = result.Scores[i]
	}

	return itemIDs, scores, nil
}

// Delete 删除数据
func (m *MilvusClient) Delete(ctx context.Context, collectionName, expr string) error {
	return m.client.Delete(ctx, collectionName, "", expr)
}

// Query 查询数据
func (m *MilvusClient) Query(
	ctx context.Context,
	collectionName string,
	expr string,
	outputFields []string,
) ([]entity.Column, error) {
	return m.client.Query(ctx, collectionName, nil, expr, outputFields)
}

