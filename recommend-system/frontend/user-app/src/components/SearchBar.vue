<script setup lang="ts">
/**
 * SearchBar.vue - 搜索栏组件
 * 
 * 功能：
 * - 输入搜索关键词
 * - 回车或点击按钮触发搜索
 * - 清空输入
 * - 加载状态显示
 * 
 * Person B 开发
 */
import { ref, watch } from 'vue'

interface Props {
  /** 搜索输入值 (v-model) */
  modelValue: string
  /** 占位文本 */
  placeholder?: string
  /** 是否正在加载 */
  loading?: boolean
  /** 是否自动聚焦 */
  autofocus?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  placeholder: '搜索电影、商品、文章...',
  loading: false,
  autofocus: false,
})

const emit = defineEmits<{
  /** 更新 v-model */
  'update:modelValue': [value: string]
  /** 触发搜索 */
  'search': [query: string]
  /** 清空输入 */
  'clear': []
}>()

// 内部输入值
const inputValue = ref(props.modelValue)

// 同步外部值变化
watch(() => props.modelValue, (val) => {
  inputValue.value = val
})

/**
 * 处理输入变化
 */
function handleInput(e: Event) {
  const value = (e.target as HTMLInputElement).value
  inputValue.value = value
  emit('update:modelValue', value)
}

/**
 * 触发搜索
 */
function handleSearch() {
  if (inputValue.value.trim()) {
    emit('search', inputValue.value.trim())
  }
}

/**
 * 处理键盘事件
 */
function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter') {
    handleSearch()
  }
}

/**
 * 清空输入
 */
function handleClear() {
  inputValue.value = ''
  emit('update:modelValue', '')
  emit('clear')
}
</script>

<template>
  <div class="search-bar">
    <div class="search-input-wrapper">
      <!-- 搜索图标 -->
      <span class="search-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="11" cy="11" r="8"></circle>
          <path d="m21 21-4.3-4.3"></path>
        </svg>
      </span>
      
      <!-- 输入框 -->
      <input
        type="text"
        class="search-input"
        :value="inputValue"
        :placeholder="placeholder"
        :autofocus="autofocus"
        @input="handleInput"
        @keydown="handleKeydown"
        data-testid="search-input"
      />
      
      <!-- 清除按钮 -->
      <button
        v-if="inputValue"
        class="clear-btn"
        @click="handleClear"
        type="button"
        aria-label="清除搜索"
        data-testid="clear-button"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M18 6 6 18"></path>
          <path d="m6 6 12 12"></path>
        </svg>
      </button>
    </div>
    
    <!-- 搜索按钮 -->
    <button
      class="search-btn"
      :disabled="loading || !inputValue.trim()"
      @click="handleSearch"
      type="button"
      data-testid="search-button"
    >
      <span v-if="loading" class="loading-spinner" data-testid="loading-spinner"></span>
      <span v-else>搜索</span>
    </button>
  </div>
</template>

<style scoped>
.search-bar {
  display: flex;
  gap: 1rem;
  max-width: 800px;
  margin: 0 auto;
}

.search-input-wrapper {
  flex: 1;
  display: flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.08);
  border: 2px solid rgba(255, 255, 255, 0.15);
  border-radius: 3rem;
  padding: 0 1.5rem;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.search-input-wrapper:focus-within {
  border-color: #6366f1;
  background: rgba(255, 255, 255, 0.12);
  box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2);
}

.search-icon {
  color: #94a3b8;
  margin-right: 0.75rem;
  display: flex;
  align-items: center;
  transition: color 0.3s;
}

.search-input-wrapper:focus-within .search-icon {
  color: #6366f1;
}

.search-input {
  flex: 1;
  padding: 1rem 0;
  background: transparent;
  border: none;
  outline: none;
  color: #f1f5f9;
  font-size: 1.1rem;
  font-family: 'Nunito', 'PingFang SC', sans-serif;
}

.search-input::placeholder {
  color: #64748b;
}

.clear-btn {
  padding: 0.5rem;
  background: transparent;
  border: none;
  color: #64748b;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
}

.clear-btn:hover {
  color: #f1f5f9;
  background: rgba(255, 255, 255, 0.1);
}

.search-btn {
  padding: 1rem 2rem;
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
  border: none;
  border-radius: 3rem;
  color: #fff;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  min-width: 100px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'Nunito', 'PingFang SC', sans-serif;
}

.search-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 10px 25px rgba(99, 102, 241, 0.4);
}

.search-btn:active:not(:disabled) {
  transform: translateY(0);
}

.search-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.loading-spinner {
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* 响应式适配 */
@media (max-width: 600px) {
  .search-bar {
    flex-direction: column;
    gap: 0.75rem;
  }
  
  .search-btn {
    width: 100%;
  }
}
</style>

