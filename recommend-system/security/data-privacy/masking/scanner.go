package masking

import (
	"regexp"
)

type Scanner struct {
	phoneRegex *regexp.Regexp
	emailRegex *regexp.Regexp
	idCardRegex *regexp.Regexp
}

func NewScanner() *Scanner {
	return &Scanner{
		phoneRegex: regexp.MustCompile(`1[3-9]\d{9}`),
		emailRegex: regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`),
		idCardRegex: regexp.MustCompile(`\d{17}[\dXx]`),
	}
}

func (s *Scanner) ScanAndMask(text string) string {
	text = s.phoneRegex.ReplaceAllStringFunc(text, MaskPhoneNumber)
	text = s.emailRegex.ReplaceAllStringFunc(text, MaskEmail)
	text = s.idCardRegex.ReplaceAllStringFunc(text, MaskIDCard)
	return text
}

