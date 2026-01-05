package security

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"testing"
	"time"
)

func TestVerifyHMACSignature(t *testing.T) {
	secret := "secret-key"
	method := "POST"
	path := "/api/v1/data"
	body := `{"foo":"bar"}`
	now := time.Now().Format(time.RFC3339)

	payload := fmt.Sprintf("%s%s%s%s", method, path, now, body)
	h := hmac.New(sha256.New, []byte(secret))
	h.Write([]byte(payload))
	validSignature := hex.EncodeToString(h.Sum(nil))

	t.Run("ValidSignature", func(t *testing.T) {
		err := VerifyHMACSignature(method, path, now, body, validSignature, secret)
		if err != nil {
			t.Errorf("Expected nil error, got %v", err)
		}
	})

	t.Run("InvalidSignature", func(t *testing.T) {
		err := VerifyHMACSignature(method, path, now, body, "invalid", secret)
		if err == nil {
			t.Error("Expected error for invalid signature")
		}
	})

	t.Run("ExpiredTimestamp", func(t *testing.T) {
		oldTime := time.Now().Add(-10 * time.Minute).Format(time.RFC3339)
		err := VerifyHMACSignature(method, path, oldTime, body, validSignature, secret)
		if err == nil {
			t.Error("Expected error for expired timestamp")
		}
	})
}

