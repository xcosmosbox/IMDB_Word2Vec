package masking

import (
	"strings"
)

// MaskPhoneNumber keeps last 4 digits
func MaskPhoneNumber(phone string) string {
	if len(phone) < 7 {
		return phone // Too short to mask safely without hiding everything
	}
	return phone[:3] + "****" + phone[len(phone)-4:]
}

// MaskEmail masks username part
func MaskEmail(email string) string {
	parts := strings.Split(email, "@")
	if len(parts) != 2 {
		return email
	}
	username := parts[0]
	domain := parts[1]
	
	if len(username) <= 1 {
		return username + "***@" + domain
	}
	
	return string(username[0]) + "***@" + domain
}

// MaskIDCard hides birthday part (simplified for 18 digits: 6 + 8 + 4)
// Standard ID: 110101 19900101 1234
func MaskIDCard(id string) string {
	if len(id) != 18 {
		return id
	}
	return id[:6] + "********" + id[14:]
}

// MaskName keeps surname
func MaskName(name string) string {
	if len(name) == 0 {
		return ""
	}
	runes := []rune(name)
	if len(runes) == 1 {
		return string(runes)
	}
	return string(runes[0]) + "**"
}

