// Package database 提供数据库连接管理
package database

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"recommend-system/pkg/logger"
	"go.uber.org/zap"
)

// PostgresDB PostgreSQL 数据库连接
type PostgresDB struct {
	pool *pgxpool.Pool
}

// PostgresConfig PostgreSQL 配置
type PostgresConfig struct {
	Host            string
	Port            int
	User            string
	Password        string
	DBName          string
	SSLMode         string
	MaxOpenConns    int32
	MaxIdleConns    int32
	ConnMaxLifetime time.Duration
	ConnMaxIdleTime time.Duration
}

// NewPostgresDB 创建 PostgreSQL 连接
func NewPostgresDB(cfg *PostgresConfig) (*PostgresDB, error) {
	dsn := fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		cfg.Host, cfg.Port, cfg.User, cfg.Password, cfg.DBName, cfg.SSLMode,
	)

	poolConfig, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to parse postgres config: %w", err)
	}

	// 连接池配置
	poolConfig.MaxConns = cfg.MaxOpenConns
	poolConfig.MinConns = cfg.MaxIdleConns
	poolConfig.MaxConnLifetime = cfg.ConnMaxLifetime
	poolConfig.MaxConnIdleTime = cfg.ConnMaxIdleTime

	// 连接钩子 - 日志
	poolConfig.BeforeAcquire = func(ctx context.Context, conn *pgx.Conn) bool {
		return true
	}

	// 创建连接池
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	pool, err := pgxpool.NewWithConfig(ctx, poolConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create postgres pool: %w", err)
	}

	// 测试连接
	if err := pool.Ping(ctx); err != nil {
		return nil, fmt.Errorf("failed to ping postgres: %w", err)
	}

	logger.Info("PostgreSQL connected",
		zap.String("host", cfg.Host),
		zap.Int("port", cfg.Port),
		zap.String("database", cfg.DBName),
	)

	return &PostgresDB{pool: pool}, nil
}

// Pool 返回连接池
func (db *PostgresDB) Pool() *pgxpool.Pool {
	return db.pool
}

// Close 关闭连接池
func (db *PostgresDB) Close() {
	if db.pool != nil {
		db.pool.Close()
		logger.Info("PostgreSQL connection closed")
	}
}

// Ping 测试连接
func (db *PostgresDB) Ping(ctx context.Context) error {
	return db.pool.Ping(ctx)
}

// QueryRow 查询单行
func (db *PostgresDB) QueryRow(ctx context.Context, sql string, args ...interface{}) pgx.Row {
	return db.pool.QueryRow(ctx, sql, args...)
}

// Query 查询多行
func (db *PostgresDB) Query(ctx context.Context, sql string, args ...interface{}) (pgx.Rows, error) {
	return db.pool.Query(ctx, sql, args...)
}

// Exec 执行 SQL
func (db *PostgresDB) Exec(ctx context.Context, sql string, args ...interface{}) error {
	_, err := db.pool.Exec(ctx, sql, args...)
	return err
}

// BeginTx 开始事务
func (db *PostgresDB) BeginTx(ctx context.Context) (pgx.Tx, error) {
	return db.pool.Begin(ctx)
}

// WithTx 在事务中执行函数
func (db *PostgresDB) WithTx(ctx context.Context, fn func(tx pgx.Tx) error) error {
	tx, err := db.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}

	defer func() {
		if p := recover(); p != nil {
			_ = tx.Rollback(ctx)
			panic(p)
		}
	}()

	if err := fn(tx); err != nil {
		if rbErr := tx.Rollback(ctx); rbErr != nil {
			return fmt.Errorf("tx err: %v, rb err: %v", err, rbErr)
		}
		return err
	}

	return tx.Commit(ctx)
}

// Stats 返回连接池统计
func (db *PostgresDB) Stats() *pgxpool.Stat {
	return db.pool.Stat()
}

