package masking

import "testing"

func TestScanner(t *testing.T) {
	s := NewScanner()
	input := "Contact: 13812345678, email: test@example.com"
	expected := "Contact: 138****5678, email: t***@example.com"

	got := s.ScanAndMask(input)
	if got != expected {
		t.Errorf("Expected '%s', got '%s'", expected, got)
	}
}

