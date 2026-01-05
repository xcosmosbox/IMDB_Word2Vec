package masking

import "testing"

func TestMasking(t *testing.T) {
	t.Run("Phone", func(t *testing.T) {
		if got := MaskPhoneNumber("13812345678"); got != "138****5678" {
			t.Errorf("Expected 138****5678, got %s", got)
		}
	})

	t.Run("Email", func(t *testing.T) {
		if got := MaskEmail("alice@example.com"); got != "a***@example.com" {
			t.Errorf("Expected a***@example.com, got %s", got)
		}
	})

	t.Run("IDCard", func(t *testing.T) {
		if got := MaskIDCard("110101199001011234"); got != "110101********1234" {
			t.Errorf("Expected 110101********1234, got %s", got)
		}
	})

	t.Run("Name", func(t *testing.T) {
		if got := MaskName("张三"); got != "张**" {
			t.Errorf("Expected 张**, got %s", got)
		}
	})
}

