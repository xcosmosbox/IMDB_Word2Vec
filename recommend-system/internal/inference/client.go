// Package inference 提供模型推理服务
package inference

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"recommend-system/internal/model"
	"recommend-system/pkg/logger"
	"go.uber.org/zap"
)

// Client 推理客户端接口
type Client interface {
	Infer(ctx context.Context, input *model.ModelInput) (*model.ModelOutput, error)
	BatchInfer(ctx context.Context, inputs []*model.ModelInput) ([]*model.ModelOutput, error)
	Health(ctx context.Context) error
}

// TritonClient Triton 推理服务客户端
type TritonClient struct {
	baseURL      string
	modelName    string
	modelVersion string
	httpClient   *http.Client
	timeout      time.Duration
}

// TritonConfig Triton 客户端配置
type TritonConfig struct {
	BaseURL      string
	ModelName    string
	ModelVersion string
	Timeout      time.Duration
}

// NewTritonClient 创建 Triton 客户端
func NewTritonClient(cfg *TritonConfig) *TritonClient {
	return &TritonClient{
		baseURL:      cfg.BaseURL,
		modelName:    cfg.ModelName,
		modelVersion: cfg.ModelVersion,
		timeout:      cfg.Timeout,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 100,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

// TritonRequest Triton 推理请求
type TritonRequest struct {
	Inputs  []TritonInput  `json:"inputs"`
	Outputs []TritonOutput `json:"outputs,omitempty"`
}

// TritonInput Triton 输入
type TritonInput struct {
	Name     string      `json:"name"`
	Shape    []int       `json:"shape"`
	Datatype string      `json:"datatype"`
	Data     interface{} `json:"data"`
}

// TritonOutput Triton 输出
type TritonOutput struct {
	Name string `json:"name"`
}

// TritonResponse Triton 响应
type TritonResponse struct {
	ModelName    string         `json:"model_name"`
	ModelVersion string         `json:"model_version"`
	Outputs      []TritonResult `json:"outputs"`
}

// TritonResult Triton 结果
type TritonResult struct {
	Name     string      `json:"name"`
	Shape    []int       `json:"shape"`
	Datatype string      `json:"datatype"`
	Data     interface{} `json:"data"`
}

// Infer 执行单次推理
func (c *TritonClient) Infer(ctx context.Context, input *model.ModelInput) (*model.ModelOutput, error) {
	// 构建请求
	request := &TritonRequest{
		Inputs: []TritonInput{
			{
				Name:     "input_ids",
				Shape:    []int{1, len(input.InputIDs)},
				Datatype: "INT64",
				Data:     [][]int64{input.InputIDs},
			},
			{
				Name:     "attention_mask",
				Shape:    []int{1, len(input.AttentionMask)},
				Datatype: "INT64",
				Data:     [][]int64{input.AttentionMask},
			},
		},
		Outputs: []TritonOutput{
			{Name: "generated_ids"},
			{Name: "logits"},
		},
	}

	// 发送请求
	resp, err := c.sendRequest(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// 解析响应
	output := &model.ModelOutput{}
	for _, result := range resp.Outputs {
		switch result.Name {
		case "generated_ids":
			if data, ok := result.Data.([]interface{}); ok {
				output.GeneratedIDs = parseInt64Array2D(data)
			}
		case "logits":
			if data, ok := result.Data.([]interface{}); ok {
				output.Logits = parseFloat32Array2D(data)
			}
		}
	}

	return output, nil
}

// BatchInfer 批量推理
func (c *TritonClient) BatchInfer(ctx context.Context, inputs []*model.ModelInput) ([]*model.ModelOutput, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	// 简化实现：逐个推理
	// 实际生产中应该支持真正的批量推理
	outputs := make([]*model.ModelOutput, len(inputs))
	for i, input := range inputs {
		output, err := c.Infer(ctx, input)
		if err != nil {
			logger.Warn("batch infer item failed",
				zap.Int("index", i),
				zap.Error(err),
			)
			continue
		}
		outputs[i] = output
	}

	return outputs, nil
}

// Health 健康检查
func (c *TritonClient) Health(ctx context.Context) error {
	url := fmt.Sprintf("%s/v2/health/ready", c.baseURL)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed: status %d", resp.StatusCode)
	}

	return nil
}

// sendRequest 发送推理请求
func (c *TritonClient) sendRequest(ctx context.Context, request *TritonRequest) (*TritonResponse, error) {
	url := fmt.Sprintf("%s/v2/models/%s/versions/%s/infer",
		c.baseURL, c.modelName, c.modelVersion)

	body, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("inference failed: status %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	var tritonResp TritonResponse
	if err := json.NewDecoder(resp.Body).Decode(&tritonResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &tritonResp, nil
}

// parseFloat32Array2D 解析二维 float32 数组
func parseFloat32Array2D(data []interface{}) [][]float32 {
	result := make([][]float32, len(data))
	for i, row := range data {
		if rowData, ok := row.([]interface{}); ok {
			result[i] = make([]float32, len(rowData))
			for j, val := range rowData {
				if f, ok := val.(float64); ok {
					result[i][j] = float32(f)
				}
			}
		}
	}
	return result
}

// parseInt64Array2D 解析二维 int64 数组
func parseInt64Array2D(data []interface{}) [][]int64 {
	result := make([][]int64, len(data))
	for i, row := range data {
		if rowData, ok := row.([]interface{}); ok {
			result[i] = make([]int64, len(rowData))
			for j, val := range rowData {
				if f, ok := val.(float64); ok {
					result[i][j] = int64(f)
				}
			}
		}
	}
	return result
}

// MockClient 模拟推理客户端 (用于测试)
type MockClient struct {
	latency time.Duration
}

// NewMockClient 创建模拟客户端
func NewMockClient(latency time.Duration) *MockClient {
	return &MockClient{latency: latency}
}

// Infer 模拟推理
func (c *MockClient) Infer(ctx context.Context, input *model.ModelInput) (*model.ModelOutput, error) {
	time.Sleep(c.latency)

	// 返回模拟数据
	return &model.ModelOutput{
		GeneratedIDs: [][]int64{
			{100, 200, 300, 400, 500},
		},
		Logits: [][]float32{
			{0.1, 0.2, 0.3, 0.2, 0.2},
		},
	}, nil
}

// BatchInfer 批量模拟推理
func (c *MockClient) BatchInfer(ctx context.Context, inputs []*model.ModelInput) ([]*model.ModelOutput, error) {
	outputs := make([]*model.ModelOutput, len(inputs))
	for i := range inputs {
		output, _ := c.Infer(ctx, inputs[i])
		outputs[i] = output
	}
	return outputs, nil
}

// Health 健康检查
func (c *MockClient) Health(ctx context.Context) error {
	return nil
}

