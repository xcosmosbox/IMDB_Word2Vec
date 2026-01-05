/**
 * useFormValidation 组合式函数单元测试
 * 
 * 测试表单验证功能：验证规则、字段验证、表单提交等。
 * 
 * @author Person C
 */

import { describe, it, expect, beforeEach } from 'vitest'
import {
  useFormValidation,
  useFieldValidation,
  required,
  email,
  minLength,
  maxLength,
  numberRange,
  passwordStrength,
  confirmPassword,
  pattern,
} from '@/composables/useFormValidation'

// =============================================================================
// 验证规则测试
// =============================================================================

describe('验证规则', () => {
  // ===========================================================================
  // required 规则
  // ===========================================================================

  describe('required', () => {
    it('空字符串应该返回错误信息', () => {
      const rule = required()
      expect(rule('')).toBe('此项为必填')
    })

    it('null 应该返回错误信息', () => {
      const rule = required()
      expect(rule(null)).toBe('此项为必填')
    })

    it('undefined 应该返回错误信息', () => {
      const rule = required()
      expect(rule(undefined)).toBe('此项为必填')
    })

    it('空数组应该返回错误信息', () => {
      const rule = required()
      expect(rule([])).toBe('此项为必填')
    })

    it('非空值应该返回 true', () => {
      const rule = required()
      expect(rule('hello')).toBe(true)
      expect(rule(0)).toBe(true)
      expect(rule(['item'])).toBe(true)
    })

    it('应该支持自定义错误信息', () => {
      const rule = required('自定义错误')
      expect(rule('')).toBe('自定义错误')
    })
  })

  // ===========================================================================
  // email 规则
  // ===========================================================================

  describe('email', () => {
    it('有效邮箱应该返回 true', () => {
      const rule = email()
      expect(rule('test@example.com')).toBe(true)
      expect(rule('user.name@domain.co.uk')).toBe(true)
      expect(rule('user+tag@example.org')).toBe(true)
    })

    it('无效邮箱应该返回错误信息', () => {
      const rule = email()
      expect(rule('invalid')).toBe('邮箱格式不正确')
      expect(rule('invalid@')).toBe('邮箱格式不正确')
      expect(rule('@domain.com')).toBe('邮箱格式不正确')
      expect(rule('user@')).toBe('邮箱格式不正确')
    })

    it('空值应该返回 true（由 required 处理）', () => {
      const rule = email()
      expect(rule('')).toBe(true)
    })

    it('应该支持自定义错误信息', () => {
      const rule = email('请输入正确的邮箱')
      expect(rule('invalid')).toBe('请输入正确的邮箱')
    })
  })

  // ===========================================================================
  // minLength 规则
  // ===========================================================================

  describe('minLength', () => {
    it('满足最小长度应该返回 true', () => {
      const rule = minLength(5)
      expect(rule('hello')).toBe(true)
      expect(rule('hello world')).toBe(true)
    })

    it('不满足最小长度应该返回错误信息', () => {
      const rule = minLength(5)
      expect(rule('hi')).toBe('至少需要 5 个字符')
    })

    it('空值应该返回 true', () => {
      const rule = minLength(5)
      expect(rule('')).toBe(true)
    })

    it('应该支持自定义错误信息', () => {
      const rule = minLength(5, '太短了')
      expect(rule('hi')).toBe('太短了')
    })
  })

  // ===========================================================================
  // maxLength 规则
  // ===========================================================================

  describe('maxLength', () => {
    it('满足最大长度应该返回 true', () => {
      const rule = maxLength(10)
      expect(rule('hello')).toBe(true)
      expect(rule('1234567890')).toBe(true)
    })

    it('超过最大长度应该返回错误信息', () => {
      const rule = maxLength(5)
      expect(rule('hello world')).toBe('最多 5 个字符')
    })

    it('应该支持自定义错误信息', () => {
      const rule = maxLength(5, '太长了')
      expect(rule('hello world')).toBe('太长了')
    })
  })

  // ===========================================================================
  // numberRange 规则
  // ===========================================================================

  describe('numberRange', () => {
    it('在范围内应该返回 true', () => {
      const rule = numberRange(1, 100)
      expect(rule(1)).toBe(true)
      expect(rule(50)).toBe(true)
      expect(rule(100)).toBe(true)
    })

    it('超出范围应该返回错误信息', () => {
      const rule = numberRange(1, 100)
      expect(rule(0)).toBe('请输入 1 到 100 之间的数字')
      expect(rule(101)).toBe('请输入 1 到 100 之间的数字')
      expect(rule(-5)).toBe('请输入 1 到 100 之间的数字')
    })

    it('null/undefined 应该返回 true', () => {
      const rule = numberRange(1, 100)
      expect(rule(null)).toBe(true)
      expect(rule(undefined)).toBe(true)
    })
  })

  // ===========================================================================
  // passwordStrength 规则
  // ===========================================================================

  describe('passwordStrength', () => {
    it('强密码应该返回 true', () => {
      const rule = passwordStrength()
      expect(rule('password123')).toBe(true)
      expect(rule('Pass1234')).toBe(true)
      expect(rule('abc123xyz')).toBe(true)
    })

    it('弱密码应该返回错误信息', () => {
      const rule = passwordStrength()
      expect(rule('12345')).toBe('密码需包含字母和数字，长度至少6位') // 太短
      expect(rule('password')).toBe('密码需包含字母和数字，长度至少6位') // 无数字
      expect(rule('123456')).toBe('密码需包含字母和数字，长度至少6位') // 无字母
    })

    it('空值应该返回 true', () => {
      const rule = passwordStrength()
      expect(rule('')).toBe(true)
    })
  })

  // ===========================================================================
  // confirmPassword 规则
  // ===========================================================================

  describe('confirmPassword', () => {
    it('密码匹配应该返回 true', () => {
      const rule = confirmPassword(() => 'mypassword123')
      expect(rule('mypassword123')).toBe(true)
    })

    it('密码不匹配应该返回错误信息', () => {
      const rule = confirmPassword(() => 'mypassword123')
      expect(rule('differentpassword')).toBe('两次密码输入不一致')
    })

    it('空值应该返回 true', () => {
      const rule = confirmPassword(() => 'mypassword123')
      expect(rule('')).toBe(true)
    })
  })

  // ===========================================================================
  // pattern 规则
  // ===========================================================================

  describe('pattern', () => {
    it('匹配正则应该返回 true', () => {
      const rule = pattern(/^\d{6}$/, '请输入6位数字')
      expect(rule('123456')).toBe(true)
    })

    it('不匹配正则应该返回错误信息', () => {
      const rule = pattern(/^\d{6}$/, '请输入6位数字')
      expect(rule('12345')).toBe('请输入6位数字')
      expect(rule('abcdef')).toBe('请输入6位数字')
    })

    it('空值应该返回 true', () => {
      const rule = pattern(/^\d{6}$/, '请输入6位数字')
      expect(rule('')).toBe(true)
    })
  })
})

// =============================================================================
// useFormValidation 组合式函数测试
// =============================================================================

describe('useFormValidation', () => {
  // ===========================================================================
  // 基础功能
  // ===========================================================================

  describe('基础功能', () => {
    it('应该初始化字段值', () => {
      const { fields } = useFormValidation({
        name: { value: '张三', rules: [] },
        email: { value: 'test@example.com', rules: [] },
      })

      expect(fields.name.value).toBe('张三')
      expect(fields.email.value).toBe('test@example.com')
    })

    it('字段初始状态应该正确', () => {
      const { fields } = useFormValidation({
        name: { value: '', rules: [required()] },
      })

      expect(fields.name.error).toBe('')
      expect(fields.name.touched).toBe(false)
    })

    it('getFormData 应该返回所有字段值', () => {
      const { fields, getFormData } = useFormValidation({
        name: { value: '张三', rules: [] },
        age: { value: 25, rules: [] },
      })

      const data = getFormData()
      expect(data.name).toBe('张三')
      expect(data.age).toBe(25)
    })
  })

  // ===========================================================================
  // 验证功能
  // ===========================================================================

  describe('验证功能', () => {
    it('validate 应该验证所有字段', () => {
      const { validate, fields } = useFormValidation({
        name: { value: '', rules: [required('请输入名称')] },
        email: { value: 'invalid', rules: [required(), email('邮箱无效')] },
      })

      const isValid = validate()

      expect(isValid).toBe(false)
      expect(fields.name.error).toBe('请输入名称')
      expect(fields.email.error).toBe('邮箱无效')
    })

    it('validate 应该标记所有字段为已触碰', () => {
      const { validate, fields } = useFormValidation({
        name: { value: '', rules: [required()] },
      })

      expect(fields.name.touched).toBe(false)

      validate()

      expect(fields.name.touched).toBe(true)
    })

    it('所有字段有效时 validate 应该返回 true', () => {
      const { validate } = useFormValidation({
        name: { value: '张三', rules: [required()] },
        email: { value: 'test@example.com', rules: [required(), email()] },
      })

      const isValid = validate()

      expect(isValid).toBe(true)
    })

    it('validateField 应该验证单个字段', () => {
      const { validateField, fields } = useFormValidation({
        name: { value: '', rules: [required('请输入名称')] },
        email: { value: 'test@example.com', rules: [email()] },
      })

      const nameValid = validateField('name')
      const emailValid = validateField('email')

      expect(nameValid).toBe(false)
      expect(fields.name.error).toBe('请输入名称')
      expect(emailValid).toBe(true)
      expect(fields.email.error).toBe('')
    })
  })

  // ===========================================================================
  // 字段操作
  // ===========================================================================

  describe('字段操作', () => {
    it('setFieldValue 应该更新字段值', () => {
      const { setFieldValue, fields } = useFormValidation({
        name: { value: '', rules: [required()] },
      })

      setFieldValue('name', '李四')

      expect(fields.name.value).toBe('李四')
    })

    it('setFieldValue 对已触碰字段应该触发验证', () => {
      const { setFieldValue, touchField, fields } = useFormValidation({
        name: { value: '', rules: [required('请输入名称')] },
      })

      touchField('name')
      expect(fields.name.error).toBe('请输入名称')

      setFieldValue('name', '张三')
      expect(fields.name.error).toBe('')
    })

    it('touchField 应该标记字段为已触碰并验证', () => {
      const { touchField, fields } = useFormValidation({
        name: { value: '', rules: [required('请输入名称')] },
      })

      expect(fields.name.touched).toBe(false)

      touchField('name')

      expect(fields.name.touched).toBe(true)
      expect(fields.name.error).toBe('请输入名称')
    })
  })

  // ===========================================================================
  // 重置功能
  // ===========================================================================

  describe('重置功能', () => {
    it('reset 应该重置所有字段到初始状态', () => {
      const { reset, fields, validate } = useFormValidation({
        name: { value: '初始值', rules: [required()] },
      })

      // 修改字段
      fields.name.value = '修改后的值'
      validate() // 触发 touched

      // 重置
      reset()

      expect(fields.name.value).toBe('初始值')
      expect(fields.name.error).toBe('')
      expect(fields.name.touched).toBe(false)
    })
  })

  // ===========================================================================
  // 计算属性
  // ===========================================================================

  describe('计算属性', () => {
    it('hasErrors 应该正确反映是否有错误', () => {
      const { hasErrors, validate, fields } = useFormValidation({
        name: { value: '', rules: [required()] },
      })

      expect(hasErrors.value).toBe(false)

      validate()

      expect(hasErrors.value).toBe(true)

      fields.name.value = '张三'
      validate()

      expect(hasErrors.value).toBe(false)
    })

    it('allTouched 应该正确反映是否所有字段都已触碰', () => {
      const { allTouched, touchField } = useFormValidation({
        name: { value: '', rules: [] },
        email: { value: '', rules: [] },
      })

      expect(allTouched.value).toBe(false)

      touchField('name')
      expect(allTouched.value).toBe(false)

      touchField('email')
      expect(allTouched.value).toBe(true)
    })

    it('isValid 应该正确反映表单是否有效', () => {
      const { isValid, validate, fields } = useFormValidation({
        name: { value: '', rules: [required()] },
      })

      expect(isValid.value).toBe(false) // 未触碰

      validate()
      expect(isValid.value).toBe(false) // 有错误

      fields.name.value = '张三'
      validate()
      expect(isValid.value).toBe(true) // 有效
    })
  })
})

// =============================================================================
// useFieldValidation 组合式函数测试
// =============================================================================

describe('useFieldValidation', () => {
  it('应该初始化值', () => {
    const field = useFieldValidation('初始值', [])

    expect(field.value.value).toBe('初始值')
    expect(field.error.value).toBe('')
    expect(field.touched.value).toBe(false)
  })

  it('validate 应该验证并更新错误', () => {
    const field = useFieldValidation('', [required('必填')])

    const isValid = field.validate()

    expect(isValid).toBe(false)
    expect(field.error.value).toBe('必填')
  })

  it('touch 应该标记为已触碰并验证', () => {
    const field = useFieldValidation('', [required('必填')])

    field.touch()

    expect(field.touched.value).toBe(true)
    expect(field.error.value).toBe('必填')
  })

  it('reset 应该重置到初始状态', () => {
    const field = useFieldValidation('初始值', [required()])

    field.value.value = '修改后'
    field.touch()

    field.reset()

    expect(field.value.value).toBe('初始值')
    expect(field.error.value).toBe('')
    expect(field.touched.value).toBe(false)
  })

  it('isValid 应该正确反映验证状态', () => {
    const field = useFieldValidation('', [required()])

    expect(field.isValid.value).toBe(false)

    field.value.value = '有值'
    field.touch()

    expect(field.isValid.value).toBe(true)
  })
})

