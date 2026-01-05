/**
 * ProfileCard - 个人信息卡片组件
 * 
 * 展示和编辑用户个人信息的卡片组件。
 * 支持查看模式和编辑模式切换。
 * 
 * @component
 * @author Person C
 */
<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { User, UpdateUserRequest } from '@shared/types'

// =============================================================================
// Props 定义
// =============================================================================

interface Props {
  /** 用户信息 */
  user: User
  /** 是否编辑模式 */
  isEditing?: boolean
  /** 加载状态 */
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  isEditing: false,
  loading: false,
})

// =============================================================================
// Emits 定义
// =============================================================================

const emit = defineEmits<{
  /** 开始编辑 */
  edit: []
  /** 保存编辑 */
  save: [data: UpdateUserRequest]
  /** 取消编辑 */
  cancel: []
}>()

// =============================================================================
// 编辑表单数据
// =============================================================================

const editForm = ref<UpdateUserRequest>({
  name: '',
  age: undefined,
  gender: '',
})

// 监听 user 变化，同步表单数据
watch(
  () => props.user,
  (newUser) => {
    if (newUser) {
      editForm.value = {
        name: newUser.name,
        age: newUser.age || undefined,
        gender: newUser.gender || '',
      }
    }
  },
  { immediate: true }
)

// =============================================================================
// 计算属性
// =============================================================================

/** 用户头像首字母 */
const avatarInitial = computed(() => {
  return props.user?.name?.charAt(0).toUpperCase() || '?'
})

/** 性别显示文本 */
const genderText = computed(() => {
  const genderMap: Record<string, string> = {
    male: '男',
    female: '女',
    other: '其他',
  }
  return genderMap[props.user?.gender] || '未设置'
})

/** 注册时间格式化 */
const formattedCreatedAt = computed(() => {
  if (!props.user?.created_at) return '未知'
  const date = new Date(props.user.created_at)
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
})

// =============================================================================
// 事件处理
// =============================================================================

/**
 * 开始编辑
 */
function handleEdit() {
  emit('edit')
}

/**
 * 保存编辑
 */
function handleSave() {
  emit('save', { ...editForm.value })
}

/**
 * 取消编辑
 */
function handleCancel() {
  // 重置表单
  if (props.user) {
    editForm.value = {
      name: props.user.name,
      age: props.user.age || undefined,
      gender: props.user.gender || '',
    }
  }
  emit('cancel')
}
</script>

<template>
  <div class="profile-card">
    <!-- 头像区域 -->
    <div class="avatar-section">
      <div class="avatar">
        <span class="avatar-text">{{ avatarInitial }}</span>
      </div>
      <div v-if="!isEditing" class="user-brief">
        <h3 class="user-name">{{ user.name }}</h3>
        <p class="user-email">{{ user.email }}</p>
      </div>
    </div>

    <!-- 查看模式 -->
    <div v-if="!isEditing" class="info-section">
      <div class="info-grid">
        <div class="info-item">
          <span class="info-label">昵称</span>
          <span class="info-value">{{ user.name }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">邮箱</span>
          <span class="info-value">{{ user.email }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">年龄</span>
          <span class="info-value">{{ user.age || '未设置' }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">性别</span>
          <span class="info-value">{{ genderText }}</span>
        </div>
        <div class="info-item full-width">
          <span class="info-label">注册时间</span>
          <span class="info-value">{{ formattedCreatedAt }}</span>
        </div>
      </div>

      <button class="edit-btn" @click="handleEdit">
        <span class="btn-icon">✏️</span>
        <span>编辑资料</span>
      </button>
    </div>

    <!-- 编辑模式 -->
    <div v-else class="edit-section">
      <div class="form-group">
        <label class="form-label">昵称</label>
        <input
          v-model="editForm.name"
          type="text"
          class="form-input"
          placeholder="请输入昵称"
        />
      </div>

      <div class="form-group">
        <label class="form-label">年龄</label>
        <input
          v-model.number="editForm.age"
          type="number"
          class="form-input"
          placeholder="请输入年龄"
          min="1"
          max="150"
        />
      </div>

      <div class="form-group">
        <label class="form-label">性别</label>
        <div class="gender-options">
          <label class="gender-option">
            <input
              v-model="editForm.gender"
              type="radio"
              value="male"
              name="gender"
            />
            <span class="gender-text">男</span>
          </label>
          <label class="gender-option">
            <input
              v-model="editForm.gender"
              type="radio"
              value="female"
              name="gender"
            />
            <span class="gender-text">女</span>
          </label>
          <label class="gender-option">
            <input
              v-model="editForm.gender"
              type="radio"
              value="other"
              name="gender"
            />
            <span class="gender-text">其他</span>
          </label>
        </div>
      </div>

      <div class="button-group">
        <button
          class="cancel-btn"
          :disabled="loading"
          @click="handleCancel"
        >
          取消
        </button>
        <button
          class="save-btn"
          :disabled="loading"
          @click="handleSave"
        >
          <span v-if="loading" class="loading-spinner"></span>
          <span>{{ loading ? '保存中...' : '保存' }}</span>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.profile-card {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 1rem;
  overflow: hidden;
}

/* 头像区域 */
.avatar-section {
  display: flex;
  align-items: center;
  gap: 1.25rem;
  padding: 1.5rem;
  background: linear-gradient(135deg, rgba(79, 172, 254, 0.15) 0%, rgba(0, 242, 254, 0.1) 100%);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.avatar {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
}

.avatar-text {
  font-size: 2rem;
  font-weight: 700;
  color: #fff;
}

.user-brief {
  flex: 1;
  min-width: 0;
}

.user-name {
  font-size: 1.5rem;
  font-weight: 700;
  color: #fff;
  margin: 0 0 0.25rem;
}

.user-email {
  font-size: 0.9rem;
  color: #8892b0;
  margin: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 信息区域 */
.info-section {
  padding: 1.5rem;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.info-item.full-width {
  grid-column: 1 / -1;
}

.info-label {
  font-size: 0.8rem;
  color: #8892b0;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.info-value {
  font-size: 1rem;
  color: #ccd6f6;
  font-weight: 500;
}

/* 编辑按钮 */
.edit-btn {
  width: 100%;
  padding: 0.875rem;
  background: rgba(79, 172, 254, 0.15);
  border: 1px solid rgba(79, 172, 254, 0.3);
  border-radius: 0.75rem;
  color: #4facfe;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  transition: all 0.3s ease;
}

.edit-btn:hover {
  background: rgba(79, 172, 254, 0.25);
  border-color: rgba(79, 172, 254, 0.5);
}

.btn-icon {
  font-size: 1rem;
}

/* 编辑区域 */
.edit-section {
  padding: 1.5rem;
}

.form-group {
  margin-bottom: 1.25rem;
}

.form-label {
  display: block;
  font-size: 0.9rem;
  color: #ccd6f6;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

.form-input {
  width: 100%;
  padding: 0.875rem 1rem;
  background: rgba(255, 255, 255, 0.05);
  border: 2px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.75rem;
  color: #fff;
  font-size: 1rem;
  transition: all 0.3s ease;
  box-sizing: border-box;
}

.form-input:focus {
  outline: none;
  border-color: #4facfe;
  box-shadow: 0 0 15px rgba(79, 172, 254, 0.2);
}

.form-input::placeholder {
  color: #8892b0;
}

/* 性别选项 */
.gender-options {
  display: flex;
  gap: 1rem;
}

.gender-option {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
}

.gender-option input[type="radio"] {
  width: 18px;
  height: 18px;
  accent-color: #4facfe;
  cursor: pointer;
}

.gender-text {
  color: #ccd6f6;
  font-size: 0.95rem;
}

/* 按钮组 */
.button-group {
  display: flex;
  gap: 1rem;
  margin-top: 1.5rem;
}

.cancel-btn,
.save-btn {
  flex: 1;
  padding: 0.875rem;
  border-radius: 0.75rem;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  transition: all 0.3s ease;
}

.cancel-btn {
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: #ccd6f6;
}

.cancel-btn:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.15);
}

.save-btn {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  border: none;
  color: #fff;
}

.save-btn:hover:not(:disabled) {
  box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
}

.cancel-btn:disabled,
.save-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* 加载动画 */
.loading-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid transparent;
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* 响应式 */
@media (max-width: 480px) {
  .info-grid {
    grid-template-columns: 1fr;
  }

  .avatar-section {
    flex-direction: column;
    text-align: center;
  }

  .gender-options {
    flex-wrap: wrap;
  }
}
</style>

