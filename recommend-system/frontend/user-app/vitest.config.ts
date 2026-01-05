/**
 * Vitest 测试配置
 * 
 * @author Person F
 */

import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  
  test: {
    // 测试环境
    environment: 'jsdom',
    
    // 全局设置
    globals: true,
    
    // 设置文件
    setupFiles: ['./src/__tests__/setup.ts'],
    
    // 包含的测试文件
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    
    // 排除的文件
    exclude: ['node_modules', 'dist'],
    
    // 覆盖率配置
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/__tests__/',
        '**/*.d.ts',
        '**/*.config.*',
      ],
    },
    
    // 超时设置
    testTimeout: 10000,
    
    // 线程
    threads: true,
  },
  
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@shared': resolve(__dirname, '../shared'),
    },
  },
})
