package crypto

import (
	"encoding/base64"
	"testing"
)

func TestCryptoService(t *testing.T) {
	// 32-byte key in base64
	keyBase64 := base64.StdEncoding.EncodeToString([]byte("12345678901234567890123456789012"))
	service, err := NewCryptoService(keyBase64)
	if err != nil {
		t.Fatalf("Failed to create service: %v", err)
	}

	t.Run("EncryptDecrypt", func(t *testing.T) {
		plaintext := []byte("Hello, World!")
		encrypted, err := service.Encrypt(plaintext)
		if err != nil {
			t.Fatalf("Encryption failed: %v", err)
		}

		decrypted, err := service.Decrypt(encrypted)
		if err != nil {
			t.Fatalf("Decryption failed: %v", err)
		}

		if string(decrypted) != string(plaintext) {
			t.Errorf("Expected %s, got %s", plaintext, decrypted)
		}
	})

	t.Run("InvalidKey", func(t *testing.T) {
		_, err := NewCryptoService("invalid")
		if err == nil {
			t.Error("Expected error for invalid key")
		}
	})
}

