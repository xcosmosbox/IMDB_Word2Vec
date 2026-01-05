/**
 * 通用工具函数单元测试
 * 
 * @author Person F
 */

import { describe, it, expect, vi } from 'vitest'
import {
  isValidEmail,
  isValidPhone,
  validatePassword,
  isValidUrl,
  debounce,
  throttle,
  sleep,
  retry,
  deepClone,
  deepMerge,
  removeEmpty,
  pick,
  omit,
  unique,
  groupBy,
  chunk,
  shuffle,
  generateId,
  uuid,
  buildQueryString,
  parseQueryString,
  isEmpty,
  isObject,
  isFunction,
  isPromise,
} from '@shared/utils'

describe('验证工具', () => {
  describe('isValidEmail', () => {
    it('应该验证正确的邮箱', () => {
      expect(isValidEmail('test@example.com')).toBe(true)
      expect(isValidEmail('user.name@domain.co')).toBe(true)
    })

    it('应该拒绝错误的邮箱', () => {
      expect(isValidEmail('invalid')).toBe(false)
      expect(isValidEmail('test@')).toBe(false)
      expect(isValidEmail('@example.com')).toBe(false)
    })
  })

  describe('isValidPhone', () => {
    it('应该验证正确的手机号', () => {
      expect(isValidPhone('13812345678')).toBe(true)
      expect(isValidPhone('15912345678')).toBe(true)
    })

    it('应该拒绝错误的手机号', () => {
      expect(isValidPhone('12345678901')).toBe(false)
      expect(isValidPhone('1381234567')).toBe(false)
      expect(isValidPhone('23812345678')).toBe(false)
    })
  })

  describe('validatePassword', () => {
    it('应该验证强密码', () => {
      const result = validatePassword('Abc12345')
      expect(result.valid).toBe(true)
    })

    it('应该拒绝太短的密码', () => {
      const result = validatePassword('Abc123')
      expect(result.valid).toBe(false)
      expect(result.message).toContain('8')
    })

    it('应该拒绝没有大写字母的密码', () => {
      const result = validatePassword('abc12345')
      expect(result.valid).toBe(false)
      expect(result.message).toContain('大写')
    })

    it('应该拒绝没有小写字母的密码', () => {
      const result = validatePassword('ABC12345')
      expect(result.valid).toBe(false)
      expect(result.message).toContain('小写')
    })

    it('应该拒绝没有数字的密码', () => {
      const result = validatePassword('Abcdefgh')
      expect(result.valid).toBe(false)
      expect(result.message).toContain('数字')
    })
  })

  describe('isValidUrl', () => {
    it('应该验证正确的 URL', () => {
      expect(isValidUrl('https://example.com')).toBe(true)
      expect(isValidUrl('http://localhost:3000')).toBe(true)
    })

    it('应该拒绝错误的 URL', () => {
      expect(isValidUrl('not-a-url')).toBe(false)
      expect(isValidUrl('example.com')).toBe(false)
    })
  })
})

describe('防抖与节流', () => {
  describe('debounce', () => {
    it('应该延迟执行函数', async () => {
      const fn = vi.fn()
      const debouncedFn = debounce(fn, 100)

      debouncedFn()
      debouncedFn()
      debouncedFn()

      expect(fn).not.toHaveBeenCalled()

      await sleep(150)

      expect(fn).toHaveBeenCalledTimes(1)
    })
  })

  describe('throttle', () => {
    it('应该限制函数执行频率', async () => {
      const fn = vi.fn()
      const throttledFn = throttle(fn, 100)

      throttledFn()
      throttledFn()
      throttledFn()

      expect(fn).toHaveBeenCalledTimes(1)

      await sleep(150)
      throttledFn()

      expect(fn).toHaveBeenCalledTimes(2)
    })
  })
})

describe('异步工具', () => {
  describe('sleep', () => {
    it('应该延迟指定时间', async () => {
      const start = Date.now()
      await sleep(100)
      const elapsed = Date.now() - start

      expect(elapsed).toBeGreaterThanOrEqual(90)
    })
  })

  describe('retry', () => {
    it('成功时应该返回结果', async () => {
      const fn = vi.fn().mockResolvedValue('success')

      const result = await retry(fn)

      expect(result).toBe('success')
      expect(fn).toHaveBeenCalledTimes(1)
    })

    it('失败后应该重试', async () => {
      const fn = vi.fn()
        .mockRejectedValueOnce(new Error('fail'))
        .mockRejectedValueOnce(new Error('fail'))
        .mockResolvedValue('success')

      const result = await retry(fn, { maxAttempts: 3, delay: 10 })

      expect(result).toBe('success')
      expect(fn).toHaveBeenCalledTimes(3)
    })

    it('超过最大重试次数应该抛出错误', async () => {
      const fn = vi.fn().mockRejectedValue(new Error('always fail'))

      await expect(retry(fn, { maxAttempts: 2, delay: 10 })).rejects.toThrow('always fail')
      expect(fn).toHaveBeenCalledTimes(2)
    })
  })
})

describe('对象工具', () => {
  describe('deepClone', () => {
    it('应该深拷贝对象', () => {
      const obj = { a: 1, b: { c: 2 } }
      const cloned = deepClone(obj)

      expect(cloned).toEqual(obj)
      expect(cloned).not.toBe(obj)
      expect(cloned.b).not.toBe(obj.b)
    })

    it('应该处理数组', () => {
      const arr = [1, [2, 3]]
      const cloned = deepClone(arr)

      expect(cloned).toEqual(arr)
      expect(cloned).not.toBe(arr)
      expect(cloned[1]).not.toBe(arr[1])
    })

    it('应该处理日期', () => {
      const date = new Date()
      const cloned = deepClone(date)

      expect(cloned.getTime()).toBe(date.getTime())
      expect(cloned).not.toBe(date)
    })
  })

  describe('deepMerge', () => {
    it('应该深度合并对象', () => {
      const target = { a: 1, b: { c: 2 } }
      const source = { b: { d: 3 }, e: 4 }

      const result = deepMerge(target, source)

      expect(result).toEqual({ a: 1, b: { c: 2, d: 3 }, e: 4 })
    })
  })

  describe('removeEmpty', () => {
    it('应该移除空值', () => {
      const obj = { a: 1, b: null, c: undefined, d: '', e: 0 }
      const result = removeEmpty(obj)

      expect(result).toEqual({ a: 1, e: 0 })
    })
  })

  describe('pick', () => {
    it('应该选取指定属性', () => {
      const obj = { a: 1, b: 2, c: 3 }
      const result = pick(obj, ['a', 'c'])

      expect(result).toEqual({ a: 1, c: 3 })
    })
  })

  describe('omit', () => {
    it('应该排除指定属性', () => {
      const obj = { a: 1, b: 2, c: 3 }
      const result = omit(obj, ['b'])

      expect(result).toEqual({ a: 1, c: 3 })
    })
  })
})

describe('数组工具', () => {
  describe('unique', () => {
    it('应该去重基本类型数组', () => {
      expect(unique([1, 2, 2, 3, 3, 3])).toEqual([1, 2, 3])
    })

    it('应该按指定 key 去重对象数组', () => {
      const arr = [
        { id: 1, name: 'a' },
        { id: 2, name: 'b' },
        { id: 1, name: 'c' },
      ]

      expect(unique(arr, 'id')).toEqual([
        { id: 1, name: 'a' },
        { id: 2, name: 'b' },
      ])
    })
  })

  describe('groupBy', () => {
    it('应该按 key 分组', () => {
      const arr = [
        { type: 'a', value: 1 },
        { type: 'b', value: 2 },
        { type: 'a', value: 3 },
      ]

      const result = groupBy(arr, 'type')

      expect(result).toEqual({
        a: [{ type: 'a', value: 1 }, { type: 'a', value: 3 }],
        b: [{ type: 'b', value: 2 }],
      })
    })
  })

  describe('chunk', () => {
    it('应该分块数组', () => {
      expect(chunk([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]])
    })
  })

  describe('shuffle', () => {
    it('应该打乱数组', () => {
      const arr = [1, 2, 3, 4, 5]
      const shuffled = shuffle(arr)

      expect(shuffled).toHaveLength(5)
      expect(shuffled.sort()).toEqual(arr)
      expect(shuffled).not.toBe(arr)
    })
  })
})

describe('ID 生成', () => {
  describe('generateId', () => {
    it('应该生成指定长度的 ID', () => {
      expect(generateId(12).length).toBe(12)
      expect(generateId(20).length).toBe(20)
    })

    it('应该生成唯一 ID', () => {
      const ids = new Set(Array.from({ length: 100 }, () => generateId()))
      expect(ids.size).toBe(100)
    })
  })

  describe('uuid', () => {
    it('应该生成有效的 UUID', () => {
      const id = uuid()
      const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/

      expect(uuidRegex.test(id)).toBe(true)
    })
  })
})

describe('URL 工具', () => {
  describe('buildQueryString', () => {
    it('应该构建查询字符串', () => {
      const params = { a: 1, b: 'test', c: null, d: undefined }
      const result = buildQueryString(params)

      expect(result).toBe('a=1&b=test')
    })
  })

  describe('parseQueryString', () => {
    it('应该解析查询字符串', () => {
      const query = 'a=1&b=test'
      const result = parseQueryString(query)

      expect(result).toEqual({ a: '1', b: 'test' })
    })
  })
})

describe('类型检查', () => {
  describe('isEmpty', () => {
    it('应该检测空值', () => {
      expect(isEmpty(null)).toBe(true)
      expect(isEmpty(undefined)).toBe(true)
      expect(isEmpty('')).toBe(true)
      expect(isEmpty('  ')).toBe(true)
      expect(isEmpty([])).toBe(true)
      expect(isEmpty({})).toBe(true)
      expect(isEmpty(0)).toBe(false)
      expect(isEmpty('text')).toBe(false)
    })
  })

  describe('isObject', () => {
    it('应该检测对象', () => {
      expect(isObject({})).toBe(true)
      expect(isObject({ a: 1 })).toBe(true)
      expect(isObject([])).toBe(false)
      expect(isObject(null)).toBe(false)
      expect(isObject('string')).toBe(false)
    })
  })

  describe('isFunction', () => {
    it('应该检测函数', () => {
      expect(isFunction(() => {})).toBe(true)
      expect(isFunction(function() {})).toBe(true)
      expect(isFunction({})).toBe(false)
    })
  })

  describe('isPromise', () => {
    it('应该检测 Promise', () => {
      expect(isPromise(Promise.resolve())).toBe(true)
      expect(isPromise(new Promise(() => {}))).toBe(true)
      expect(isPromise({})).toBe(false)
    })
  })
})

