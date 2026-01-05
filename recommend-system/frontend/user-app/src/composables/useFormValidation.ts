/**
 * useFormValidation - 表单验证组合式函数
 * 
 * 提供通用的表单验证逻辑，支持常见的验证规则。
 * 
 * @module composables/useFormValidation
 * @author Person C
 */

import { ref, reactive, computed } from 'vue'

// =============================================================================
// 类型定义
// =============================================================================

/** 验证规则函数 */
export type ValidationRule = (value: any) => string | true

/** 验证结果 */
export interface ValidationResult {
  valid: boolean
  message: string
}

/** 字段验证状态 */
export interface FieldValidation {
  value: any
  error: string
  touched: boolean
  validate: () => boolean
  reset: () => void
}

// =============================================================================
// 预定义验证规则
// =============================================================================

/**
 * 必填验证
 */
export const required = (message = '此项为必填'): ValidationRule => {
  return (value: any) => {
    if (value === null || value === undefined || value === '') {
      return message
    }
    if (Array.isArray(value) && value.length === 0) {
      return message
    }
    return true
  }
}

/**
 * 邮箱格式验证
 */
export const email = (message = '邮箱格式不正确'): ValidationRule => {
  return (value: string) => {
    if (!value) return true // 空值由 required 处理
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    return emailRegex.test(value) || message
  }
}

/**
 * 最小长度验证
 */
export const minLength = (min: number, message?: string): ValidationRule => {
  return (value: string) => {
    if (!value) return true // 空值由 required 处理
    const msg = message || `至少需要 ${min} 个字符`
    return value.length >= min || msg
  }
}

/**
 * 最大长度验证
 */
export const maxLength = (max: number, message?: string): ValidationRule => {
  return (value: string) => {
    if (!value) return true
    const msg = message || `最多 ${max} 个字符`
    return value.length <= max || msg
  }
}

/**
 * 数字范围验证
 */
export const numberRange = (
  min: number,
  max: number,
  message?: string
): ValidationRule => {
  return (value: number) => {
    if (value === null || value === undefined) return true
    const msg = message || `请输入 ${min} 到 ${max} 之间的数字`
    return (value >= min && value <= max) || msg
  }
}

/**
 * 密码强度验证
 */
export const passwordStrength = (message?: string): ValidationRule => {
  return (value: string) => {
    if (!value) return true
    const msg = message || '密码需包含字母和数字，长度至少6位'
    const hasLetter = /[a-zA-Z]/.test(value)
    const hasNumber = /\d/.test(value)
    return (value.length >= 6 && hasLetter && hasNumber) || msg
  }
}

/**
 * 确认密码验证
 */
export const confirmPassword = (
  getPassword: () => string,
  message = '两次密码输入不一致'
): ValidationRule => {
  return (value: string) => {
    if (!value) return true
    return value === getPassword() || message
  }
}

/**
 * 正则表达式验证
 */
export const pattern = (regex: RegExp, message: string): ValidationRule => {
  return (value: string) => {
    if (!value) return true
    return regex.test(value) || message
  }
}

// =============================================================================
// 组合式函数
// =============================================================================

/**
 * 表单验证组合式函数
 * 
 * @example
 * ```typescript
 * const { fields, validate, reset, isValid } = useFormValidation({
 *   email: {
 *     value: '',
 *     rules: [required(), email()]
 *   },
 *   password: {
 *     value: '',
 *     rules: [required(), minLength(6)]
 *   }
 * })
 * 
 * // 使用
 * fields.email.value = 'user@example.com'
 * const valid = validate()
 * ```
 */
export function useFormValidation<
  T extends Record<string, { value: any; rules: ValidationRule[] }>
>(config: T) {
  // 字段状态
  type FieldKey = keyof T
  type FieldState = {
    [K in FieldKey]: {
      value: T[K]['value']
      error: string
      touched: boolean
    }
  }

  const fields = reactive<FieldState>(
    Object.fromEntries(
      Object.entries(config).map(([key, { value }]) => [
        key,
        { value, error: '', touched: false },
      ])
    ) as FieldState
  )

  /**
   * 验证单个字段
   */
  function validateField(key: FieldKey): boolean {
    const field = fields[key]
    const rules = config[key].rules

    for (const rule of rules) {
      const result = rule(field.value)
      if (result !== true) {
        field.error = result
        return false
      }
    }

    field.error = ''
    return true
  }

  /**
   * 验证所有字段
   */
  function validate(): boolean {
    let isValid = true
    for (const key of Object.keys(config) as FieldKey[]) {
      fields[key].touched = true
      if (!validateField(key)) {
        isValid = false
      }
    }
    return isValid
  }

  /**
   * 重置表单
   */
  function reset() {
    for (const key of Object.keys(config) as FieldKey[]) {
      fields[key].value = config[key].value
      fields[key].error = ''
      fields[key].touched = false
    }
  }

  /**
   * 设置字段值
   */
  function setFieldValue(key: FieldKey, value: any) {
    fields[key].value = value
    if (fields[key].touched) {
      validateField(key)
    }
  }

  /**
   * 触碰字段
   */
  function touchField(key: FieldKey) {
    fields[key].touched = true
    validateField(key)
  }

  /**
   * 获取表单数据
   */
  function getFormData(): Record<FieldKey, any> {
    return Object.fromEntries(
      Object.entries(fields).map(([key, field]: [string, any]) => [key, field.value])
    ) as Record<FieldKey, any>
  }

  /**
   * 是否有错误
   */
  const hasErrors = computed(() => {
    return Object.values(fields).some((field: any) => field.error)
  })

  /**
   * 是否所有字段都已触碰
   */
  const allTouched = computed(() => {
    return Object.values(fields).every((field: any) => field.touched)
  })

  /**
   * 表单是否有效
   */
  const isValid = computed(() => {
    return allTouched.value && !hasErrors.value
  })

  return {
    fields,
    validate,
    validateField,
    reset,
    setFieldValue,
    touchField,
    getFormData,
    hasErrors,
    allTouched,
    isValid,
  }
}

// =============================================================================
// 简单表单验证钩子
// =============================================================================

/**
 * 简单字段验证
 * 
 * @example
 * ```typescript
 * const emailValidation = useFieldValidation('', [required(), email()])
 * 
 * // 在模板中使用
 * <input v-model="emailValidation.value" @blur="emailValidation.touch()" />
 * <span v-if="emailValidation.error">{{ emailValidation.error }}</span>
 * ```
 */
export function useFieldValidation(initialValue: any, rules: ValidationRule[]) {
  const value = ref(initialValue)
  const error = ref('')
  const touched = ref(false)

  /**
   * 验证字段
   */
  function validate(): boolean {
    for (const rule of rules) {
      const result = rule(value.value)
      if (result !== true) {
        error.value = result
        return false
      }
    }
    error.value = ''
    return true
  }

  /**
   * 触碰字段
   */
  function touch() {
    touched.value = true
    validate()
  }

  /**
   * 重置字段
   */
  function reset() {
    value.value = initialValue
    error.value = ''
    touched.value = false
  }

  /**
   * 是否有效
   */
  const isValid = computed(() => {
    return touched.value && !error.value
  })

  return {
    value,
    error,
    touched,
    validate,
    touch,
    reset,
    isValid,
  }
}

// 导出类型
export type UseFormValidation = ReturnType<typeof useFormValidation>
export type UseFieldValidation = ReturnType<typeof useFieldValidation>

