// Package utils 提供通用工具函数
package utils

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"math"
	"sort"
	"time"
)

// GenerateID 生成唯一 ID
func GenerateID(prefix string) string {
	bytes := make([]byte, 16)
	rand.Read(bytes)
	return prefix + hex.EncodeToString(bytes)
}

// GenerateRequestID 生成请求 ID
func GenerateRequestID() string {
	return GenerateID("req_")
}

// GenerateTraceID 生成追踪 ID
func GenerateTraceID() string {
	return GenerateID("trace_")
}

// MinInt 返回两个整数的最小值
func MinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// MaxInt 返回两个整数的最大值
func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// MinFloat64 返回两个浮点数的最小值
func MinFloat64(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// MaxFloat64 返回两个浮点数的最大值
func MaxFloat64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

// Contains 检查切片是否包含元素
func Contains[T comparable](slice []T, item T) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// Unique 去重
func Unique[T comparable](slice []T) []T {
	seen := make(map[T]bool)
	result := make([]T, 0, len(slice))
	for _, item := range slice {
		if !seen[item] {
			seen[item] = true
			result = append(result, item)
		}
	}
	return result
}

// Difference 返回在 a 中但不在 b 中的元素
func Difference[T comparable](a, b []T) []T {
	bSet := make(map[T]bool)
	for _, item := range b {
		bSet[item] = true
	}

	result := make([]T, 0)
	for _, item := range a {
		if !bSet[item] {
			result = append(result, item)
		}
	}
	return result
}

// Intersection 返回两个切片的交集
func Intersection[T comparable](a, b []T) []T {
	bSet := make(map[T]bool)
	for _, item := range b {
		bSet[item] = true
	}

	result := make([]T, 0)
	for _, item := range a {
		if bSet[item] {
			result = append(result, item)
		}
	}
	return result
}

// Chunk 将切片分成指定大小的块
func Chunk[T any](slice []T, size int) [][]T {
	if size <= 0 {
		return nil
	}

	var chunks [][]T
	for i := 0; i < len(slice); i += size {
		end := i + size
		if end > len(slice) {
			end = len(slice)
		}
		chunks = append(chunks, slice[i:end])
	}
	return chunks
}

// ToJSON 转换为 JSON 字符串
func ToJSON(v interface{}) string {
	bytes, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(bytes)
}

// FromJSON 从 JSON 字符串解析
func FromJSON[T any](s string) (T, error) {
	var result T
	err := json.Unmarshal([]byte(s), &result)
	return result, err
}

// CosineSimilarity 计算余弦相似度
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// DotProduct 计算点积
func DotProduct(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var result float64
	for i := range a {
		result += float64(a[i]) * float64(b[i])
	}
	return result
}

// Normalize 向量归一化
func Normalize(v []float32) []float32 {
	var norm float64
	for _, val := range v {
		norm += float64(val) * float64(val)
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		return v
	}

	result := make([]float32, len(v))
	for i, val := range v {
		result[i] = float32(float64(val) / norm)
	}
	return result
}

// Softmax 计算 Softmax
func Softmax(logits []float64) []float64 {
	// 找最大值以提高数值稳定性
	maxLogit := logits[0]
	for _, l := range logits {
		if l > maxLogit {
			maxLogit = l
		}
	}

	// 计算 exp 和 sum
	exps := make([]float64, len(logits))
	var sum float64
	for i, l := range logits {
		exps[i] = math.Exp(l - maxLogit)
		sum += exps[i]
	}

	// 归一化
	result := make([]float64, len(logits))
	for i := range exps {
		result[i] = exps[i] / sum
	}
	return result
}

// TopK 返回 Top-K 索引和值
func TopK(values []float64, k int) ([]int, []float64) {
	if k <= 0 || len(values) == 0 {
		return nil, nil
	}

	if k > len(values) {
		k = len(values)
	}

	// 创建索引数组
	indices := make([]int, len(values))
	for i := range indices {
		indices[i] = i
	}

	// 按值降序排序索引
	sort.Slice(indices, func(i, j int) bool {
		return values[indices[i]] > values[indices[j]]
	})

	// 返回 Top-K
	topIndices := indices[:k]
	topValues := make([]float64, k)
	for i, idx := range topIndices {
		topValues[i] = values[idx]
	}

	return topIndices, topValues
}

// TimeSlot 获取时间槽
func TimeSlot(t time.Time) string {
	hour := t.Hour()
	switch {
	case hour >= 0 && hour < 6:
		return "night"
	case hour >= 6 && hour < 12:
		return "morning"
	case hour >= 12 && hour < 18:
		return "afternoon"
	default:
		return "evening"
	}
}

// DayOfWeek 获取星期类型
func DayOfWeek(t time.Time) string {
	day := t.Weekday()
	if day == time.Saturday || day == time.Sunday {
		return "weekend"
	}
	return "weekday"
}

// Retry 重试函数
func Retry(attempts int, sleep time.Duration, f func() error) error {
	var err error
	for i := 0; i < attempts; i++ {
		if err = f(); err == nil {
			return nil
		}
		time.Sleep(sleep)
		sleep *= 2 // 指数退避
	}
	return err
}

// Timer 计时器
type Timer struct {
	start time.Time
}

// NewTimer 创建计时器
func NewTimer() *Timer {
	return &Timer{start: time.Now()}
}

// Elapsed 返回经过的时间（毫秒）
func (t *Timer) Elapsed() int64 {
	return time.Since(t.start).Milliseconds()
}

// ElapsedDuration 返回经过的时间
func (t *Timer) ElapsedDuration() time.Duration {
	return time.Since(t.start)
}

