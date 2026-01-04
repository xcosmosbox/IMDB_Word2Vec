// Package mocks 提供测试用的 Mock 实现
//
// 本包实现了 interfaces 包中定义的所有接口的 Mock 版本，
// 用于单元测试和集成测试时隔离外部依赖。
package mocks

import "errors"

// =============================================================================
// 通用错误定义
// =============================================================================

var (
	// ErrNotFound 资源不存在
	ErrNotFound = errors.New("not found")

	// ErrCacheMiss 缓存未命中
	ErrCacheMiss = errors.New("cache miss")

	// ErrInvalidInput 无效输入
	ErrInvalidInput = errors.New("invalid input")

	// ErrDuplicate 重复数据
	ErrDuplicate = errors.New("duplicate entry")

	// ErrTimeout 操作超时
	ErrTimeout = errors.New("operation timeout")

	// ErrServiceUnavailable 服务不可用
	ErrServiceUnavailable = errors.New("service unavailable")
)

