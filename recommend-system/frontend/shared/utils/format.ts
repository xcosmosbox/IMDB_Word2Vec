/**
 * 格式化工具模块
 * 
 * 提供各种数据格式化功能，包括日期、数字、文本等
 * 
 * @module shared/utils/format
 * @author Person F
 */

// =============================================================================
// 日期格式化
// =============================================================================

/**
 * 格式化日期
 * 
 * @param date - 日期对象或字符串
 * @param format - 格式模板
 * @returns 格式化后的日期字符串
 * 
 * @example
 * ```typescript
 * formatDate(new Date(), 'YYYY-MM-DD') // '2024-01-15'
 * formatDate('2024-01-15T10:30:00', 'YYYY-MM-DD HH:mm') // '2024-01-15 10:30'
 * ```
 */
export function formatDate(
  date: Date | string | number,
  format = 'YYYY-MM-DD HH:mm:ss'
): string {
  const d = typeof date === 'string' || typeof date === 'number' ? new Date(date) : date
  
  if (isNaN(d.getTime())) {
    return ''
  }
  
  const tokens: Record<string, string> = {
    YYYY: String(d.getFullYear()),
    YY: String(d.getFullYear()).slice(-2),
    MM: String(d.getMonth() + 1).padStart(2, '0'),
    M: String(d.getMonth() + 1),
    DD: String(d.getDate()).padStart(2, '0'),
    D: String(d.getDate()),
    HH: String(d.getHours()).padStart(2, '0'),
    H: String(d.getHours()),
    mm: String(d.getMinutes()).padStart(2, '0'),
    m: String(d.getMinutes()),
    ss: String(d.getSeconds()).padStart(2, '0'),
    s: String(d.getSeconds()),
  }
  
  let result = format
  for (const [token, value] of Object.entries(tokens)) {
    result = result.replace(new RegExp(token, 'g'), value)
  }
  
  return result
}

/**
 * 相对时间格式化
 * 
 * @param date - 日期对象或字符串
 * @returns 相对时间字符串
 * 
 * @example
 * ```typescript
 * formatRelativeTime(new Date(Date.now() - 60000)) // '1 分钟前'
 * formatRelativeTime(new Date(Date.now() - 3600000)) // '1 小时前'
 * ```
 */
export function formatRelativeTime(date: Date | string | number): string {
  const d = typeof date === 'string' || typeof date === 'number' ? new Date(date) : date
  const now = new Date()
  const diff = now.getTime() - d.getTime()
  
  if (diff < 0) {
    return '刚刚'
  }
  
  const seconds = Math.floor(diff / 1000)
  const minutes = Math.floor(seconds / 60)
  const hours = Math.floor(minutes / 60)
  const days = Math.floor(hours / 24)
  const months = Math.floor(days / 30)
  const years = Math.floor(days / 365)
  
  if (years > 0) return `${years} 年前`
  if (months > 0) return `${months} 个月前`
  if (days > 0) return `${days} 天前`
  if (hours > 0) return `${hours} 小时前`
  if (minutes > 0) return `${minutes} 分钟前`
  if (seconds > 10) return `${seconds} 秒前`
  return '刚刚'
}

/**
 * 格式化时间范围
 * 
 * @param start - 开始日期
 * @param end - 结束日期
 * @returns 时间范围字符串
 */
export function formatDateRange(
  start: Date | string,
  end: Date | string,
  format = 'MM-DD'
): string {
  return `${formatDate(start, format)} ~ ${formatDate(end, format)}`
}

/**
 * 格式化持续时间（秒）
 * 
 * @param seconds - 秒数
 * @returns 格式化的持续时间
 * 
 * @example
 * ```typescript
 * formatDuration(3661) // '1:01:01'
 * formatDuration(125) // '2:05'
 * ```
 */
export function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  
  if (h > 0) {
    return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  }
  return `${m}:${String(s).padStart(2, '0')}`
}

// =============================================================================
// 数字格式化
// =============================================================================

/**
 * 格式化数字（添加千位分隔符）
 * 
 * @param num - 数字
 * @param decimals - 小数位数
 * @returns 格式化后的数字字符串
 * 
 * @example
 * ```typescript
 * formatNumber(1234567) // '1,234,567'
 * formatNumber(1234.5678, 2) // '1,234.57'
 * ```
 */
export function formatNumber(num: number, decimals?: number): string {
  if (isNaN(num)) return '0'
  
  const value = decimals !== undefined ? num.toFixed(decimals) : String(num)
  const parts = value.split('.')
  parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',')
  
  return parts.join('.')
}

/**
 * 格式化大数字（如：1.2万、3.5亿）
 * 
 * @param num - 数字
 * @param decimals - 小数位数
 * @returns 格式化后的字符串
 * 
 * @example
 * ```typescript
 * formatLargeNumber(12345) // '1.2万'
 * formatLargeNumber(123456789) // '1.23亿'
 * ```
 */
export function formatLargeNumber(num: number, decimals = 1): string {
  if (isNaN(num)) return '0'
  
  const absNum = Math.abs(num)
  const sign = num < 0 ? '-' : ''
  
  if (absNum >= 100000000) {
    return `${sign}${(absNum / 100000000).toFixed(decimals)}亿`
  }
  if (absNum >= 10000) {
    return `${sign}${(absNum / 10000).toFixed(decimals)}万`
  }
  if (absNum >= 1000) {
    return `${sign}${formatNumber(absNum, 0)}`
  }
  
  return `${sign}${absNum}`
}

/**
 * 格式化百分比
 * 
 * @param value - 小数值（0-1）
 * @param decimals - 小数位数
 * @returns 百分比字符串
 * 
 * @example
 * ```typescript
 * formatPercent(0.1234) // '12.34%'
 * formatPercent(0.5, 0) // '50%'
 * ```
 */
export function formatPercent(value: number, decimals = 2): string {
  if (isNaN(value)) return '0%'
  return `${(value * 100).toFixed(decimals)}%`
}

/**
 * 格式化金额
 * 
 * @param amount - 金额
 * @param currency - 货币符号
 * @param decimals - 小数位数
 * @returns 格式化后的金额字符串
 * 
 * @example
 * ```typescript
 * formatMoney(1234.5) // '¥1,234.50'
 * formatMoney(1234.5, '$') // '$1,234.50'
 * ```
 */
export function formatMoney(
  amount: number,
  currency = '¥',
  decimals = 2
): string {
  if (isNaN(amount)) return `${currency}0.00`
  return `${currency}${formatNumber(amount, decimals)}`
}

/**
 * 格式化文件大小
 * 
 * @param bytes - 字节数
 * @param decimals - 小数位数
 * @returns 格式化后的文件大小
 * 
 * @example
 * ```typescript
 * formatFileSize(1024) // '1 KB'
 * formatFileSize(1234567890) // '1.15 GB'
 * ```
 */
export function formatFileSize(bytes: number, decimals = 2): string {
  if (bytes === 0) return '0 B'
  
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`
}

// =============================================================================
// 文本格式化
// =============================================================================

/**
 * 截断文本
 * 
 * @param text - 原始文本
 * @param maxLength - 最大长度
 * @param suffix - 截断后缀
 * @returns 截断后的文本
 * 
 * @example
 * ```typescript
 * truncate('这是一段很长的文本', 5) // '这是一段...'
 * truncate('短文本', 10) // '短文本'
 * ```
 */
export function truncate(text: string, maxLength: number, suffix = '...'): string {
  if (!text || text.length <= maxLength) return text
  return text.slice(0, maxLength) + suffix
}

/**
 * 首字母大写
 */
export function capitalize(text: string): string {
  if (!text) return ''
  return text.charAt(0).toUpperCase() + text.slice(1)
}

/**
 * 转换为驼峰命名
 */
export function toCamelCase(text: string): string {
  return text
    .replace(/[-_\s]+(.)?/g, (_, c) => (c ? c.toUpperCase() : ''))
    .replace(/^./, s => s.toLowerCase())
}

/**
 * 转换为短横线命名
 */
export function toKebabCase(text: string): string {
  return text
    .replace(/([a-z])([A-Z])/g, '$1-$2')
    .replace(/[\s_]+/g, '-')
    .toLowerCase()
}

/**
 * 转换为下划线命名
 */
export function toSnakeCase(text: string): string {
  return text
    .replace(/([a-z])([A-Z])/g, '$1_$2')
    .replace(/[-\s]+/g, '_')
    .toLowerCase()
}

/**
 * 脱敏手机号
 * 
 * @example
 * ```typescript
 * maskPhone('13812345678') // '138****5678'
 * ```
 */
export function maskPhone(phone: string): string {
  if (!phone || phone.length < 7) return phone
  return phone.replace(/(\d{3})\d{4}(\d+)/, '$1****$2')
}

/**
 * 脱敏邮箱
 * 
 * @example
 * ```typescript
 * maskEmail('test@example.com') // 'te***@example.com'
 * ```
 */
export function maskEmail(email: string): string {
  if (!email || !email.includes('@')) return email
  const [name, domain] = email.split('@')
  if (name.length <= 2) return email
  return `${name.slice(0, 2)}***@${domain}`
}

/**
 * 脱敏身份证号
 * 
 * @example
 * ```typescript
 * maskIdCard('110101199001011234') // '110101********1234'
 * ```
 */
export function maskIdCard(idCard: string): string {
  if (!idCard || idCard.length < 15) return idCard
  return idCard.replace(/(\d{6})\d+(\d{4})/, '$1********$2')
}

// =============================================================================
// 物品类型格式化
// =============================================================================

/**
 * 物品类型映射
 */
const ITEM_TYPE_MAP: Record<string, string> = {
  movie: '电影',
  product: '商品',
  article: '文章',
  video: '视频',
}

/**
 * 格式化物品类型
 */
export function formatItemType(type: string): string {
  return ITEM_TYPE_MAP[type] || type
}

/**
 * 用户行为映射
 */
const ACTION_MAP: Record<string, string> = {
  view: '浏览',
  click: '点击',
  like: '喜欢',
  dislike: '不喜欢',
  buy: '购买',
  share: '分享',
}

/**
 * 格式化用户行为
 */
export function formatAction(action: string): string {
  return ACTION_MAP[action] || action
}

/**
 * 性别映射
 */
const GENDER_MAP: Record<string, string> = {
  male: '男',
  female: '女',
  unknown: '未知',
}

/**
 * 格式化性别
 */
export function formatGender(gender: string): string {
  return GENDER_MAP[gender] || gender
}

// =============================================================================
// 评分格式化
// =============================================================================

/**
 * 格式化评分
 * 
 * @param rating - 评分值（0-10）
 * @param max - 最大分数
 * @returns 格式化后的评分
 */
export function formatRating(rating: number, max = 10): string {
  if (isNaN(rating)) return '-'
  return `${rating.toFixed(1)}/${max}`
}

/**
 * 评分转星级
 * 
 * @param rating - 评分值（0-10）
 * @param maxStars - 最大星数
 * @returns 星级数
 */
export function ratingToStars(rating: number, maxStars = 5): number {
  if (isNaN(rating)) return 0
  return Math.round((rating / 10) * maxStars * 2) / 2
}

