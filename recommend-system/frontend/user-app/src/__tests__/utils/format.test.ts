/**
 * 格式化工具函数单元测试
 * 
 * @author Person F
 */

import { describe, it, expect } from 'vitest'
import {
  formatDate,
  formatRelativeTime,
  formatDuration,
  formatNumber,
  formatLargeNumber,
  formatPercent,
  formatMoney,
  formatFileSize,
  truncate,
  capitalize,
  toCamelCase,
  toKebabCase,
  toSnakeCase,
  maskPhone,
  maskEmail,
  maskIdCard,
  formatItemType,
  formatAction,
  formatGender,
  formatRating,
  ratingToStars,
} from '@shared/utils'

describe('日期格式化', () => {
  describe('formatDate', () => {
    it('应该正确格式化日期', () => {
      const date = new Date('2024-01-15T10:30:45')

      expect(formatDate(date, 'YYYY-MM-DD')).toBe('2024-01-15')
      expect(formatDate(date, 'YYYY/MM/DD')).toBe('2024/01/15')
      expect(formatDate(date, 'HH:mm:ss')).toBe('10:30:45')
      expect(formatDate(date, 'YYYY-MM-DD HH:mm:ss')).toBe('2024-01-15 10:30:45')
    })

    it('应该处理字符串日期', () => {
      expect(formatDate('2024-01-15', 'YYYY-MM-DD')).toBe('2024-01-15')
    })

    it('应该处理时间戳', () => {
      const timestamp = new Date('2024-01-15').getTime()
      expect(formatDate(timestamp, 'YYYY-MM-DD')).toBe('2024-01-15')
    })

    it('无效日期应该返回空字符串', () => {
      expect(formatDate('invalid', 'YYYY-MM-DD')).toBe('')
    })
  })

  describe('formatRelativeTime', () => {
    it('应该显示刚刚', () => {
      const now = new Date()
      expect(formatRelativeTime(now)).toBe('刚刚')
    })

    it('应该显示分钟前', () => {
      const date = new Date(Date.now() - 5 * 60 * 1000)
      expect(formatRelativeTime(date)).toBe('5 分钟前')
    })

    it('应该显示小时前', () => {
      const date = new Date(Date.now() - 3 * 60 * 60 * 1000)
      expect(formatRelativeTime(date)).toBe('3 小时前')
    })

    it('应该显示天前', () => {
      const date = new Date(Date.now() - 2 * 24 * 60 * 60 * 1000)
      expect(formatRelativeTime(date)).toBe('2 天前')
    })
  })

  describe('formatDuration', () => {
    it('应该格式化秒数', () => {
      expect(formatDuration(65)).toBe('1:05')
      expect(formatDuration(3661)).toBe('1:01:01')
      expect(formatDuration(0)).toBe('0:00')
    })
  })
})

describe('数字格式化', () => {
  describe('formatNumber', () => {
    it('应该添加千位分隔符', () => {
      expect(formatNumber(1234567)).toBe('1,234,567')
      expect(formatNumber(1000)).toBe('1,000')
      expect(formatNumber(100)).toBe('100')
    })

    it('应该保留指定小数位', () => {
      expect(formatNumber(1234.5678, 2)).toBe('1,234.57')
      expect(formatNumber(1000, 2)).toBe('1,000.00')
    })
  })

  describe('formatLargeNumber', () => {
    it('应该格式化大数字', () => {
      expect(formatLargeNumber(12345)).toBe('1.2万')
      expect(formatLargeNumber(123456789)).toBe('1.2亿')
      expect(formatLargeNumber(999)).toBe('999')
    })

    it('应该处理负数', () => {
      expect(formatLargeNumber(-12345)).toBe('-1.2万')
    })
  })

  describe('formatPercent', () => {
    it('应该格式化百分比', () => {
      expect(formatPercent(0.1234)).toBe('12.34%')
      expect(formatPercent(0.5, 0)).toBe('50%')
      expect(formatPercent(1)).toBe('100.00%')
    })
  })

  describe('formatMoney', () => {
    it('应该格式化金额', () => {
      expect(formatMoney(1234.5)).toBe('¥1,234.50')
      expect(formatMoney(1234.5, '$')).toBe('$1,234.50')
      expect(formatMoney(1000, '€', 0)).toBe('€1,000')
    })
  })

  describe('formatFileSize', () => {
    it('应该格式化文件大小', () => {
      expect(formatFileSize(0)).toBe('0 B')
      expect(formatFileSize(1024)).toBe('1 KB')
      expect(formatFileSize(1048576)).toBe('1 MB')
      expect(formatFileSize(1073741824)).toBe('1 GB')
    })
  })
})

describe('文本格式化', () => {
  describe('truncate', () => {
    it('应该截断长文本', () => {
      expect(truncate('这是一段很长的文本', 5)).toBe('这是一段很...')
      expect(truncate('短文本', 10)).toBe('短文本')
    })

    it('应该支持自定义后缀', () => {
      expect(truncate('这是一段很长的文本', 5, '…')).toBe('这是一段很…')
    })
  })

  describe('capitalize', () => {
    it('应该首字母大写', () => {
      expect(capitalize('hello')).toBe('Hello')
      expect(capitalize('WORLD')).toBe('WORLD')
      expect(capitalize('')).toBe('')
    })
  })

  describe('toCamelCase', () => {
    it('应该转换为驼峰命名', () => {
      expect(toCamelCase('hello-world')).toBe('helloWorld')
      expect(toCamelCase('hello_world')).toBe('helloWorld')
      expect(toCamelCase('hello world')).toBe('helloWorld')
    })
  })

  describe('toKebabCase', () => {
    it('应该转换为短横线命名', () => {
      expect(toKebabCase('helloWorld')).toBe('hello-world')
      expect(toKebabCase('HelloWorld')).toBe('hello-world')
    })
  })

  describe('toSnakeCase', () => {
    it('应该转换为下划线命名', () => {
      expect(toSnakeCase('helloWorld')).toBe('hello_world')
      expect(toSnakeCase('HelloWorld')).toBe('hello_world')
    })
  })

  describe('maskPhone', () => {
    it('应该脱敏手机号', () => {
      expect(maskPhone('13812345678')).toBe('138****5678')
      expect(maskPhone('123')).toBe('123')
    })
  })

  describe('maskEmail', () => {
    it('应该脱敏邮箱', () => {
      expect(maskEmail('test@example.com')).toBe('te***@example.com')
      expect(maskEmail('ab@test.com')).toBe('ab@test.com')
    })
  })

  describe('maskIdCard', () => {
    it('应该脱敏身份证号', () => {
      expect(maskIdCard('110101199001011234')).toBe('110101********1234')
      expect(maskIdCard('123')).toBe('123')
    })
  })
})

describe('业务格式化', () => {
  describe('formatItemType', () => {
    it('应该格式化物品类型', () => {
      expect(formatItemType('movie')).toBe('电影')
      expect(formatItemType('product')).toBe('商品')
      expect(formatItemType('article')).toBe('文章')
      expect(formatItemType('video')).toBe('视频')
      expect(formatItemType('unknown')).toBe('unknown')
    })
  })

  describe('formatAction', () => {
    it('应该格式化用户行为', () => {
      expect(formatAction('view')).toBe('浏览')
      expect(formatAction('click')).toBe('点击')
      expect(formatAction('like')).toBe('喜欢')
      expect(formatAction('buy')).toBe('购买')
    })
  })

  describe('formatGender', () => {
    it('应该格式化性别', () => {
      expect(formatGender('male')).toBe('男')
      expect(formatGender('female')).toBe('女')
      expect(formatGender('unknown')).toBe('未知')
    })
  })

  describe('formatRating', () => {
    it('应该格式化评分', () => {
      expect(formatRating(8.5)).toBe('8.5/10')
      expect(formatRating(9.0, 5)).toBe('9.0/5')
    })
  })

  describe('ratingToStars', () => {
    it('应该转换为星级', () => {
      expect(ratingToStars(10)).toBe(5)
      expect(ratingToStars(8)).toBe(4)
      expect(ratingToStars(5)).toBe(2.5)
      expect(ratingToStars(0)).toBe(0)
    })
  })
})

