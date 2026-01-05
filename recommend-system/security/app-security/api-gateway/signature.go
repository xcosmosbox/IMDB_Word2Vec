package security

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"time"
)

// VerifyHMACSignature validates request signature
// Signature = HMAC-SHA256(SecretKey, Method + Path + Timestamp + Body)
func VerifyHMACSignature(method, path, timestamp, body, signature, secret string) error {
	// 1. Verify timestamp (prevent replay, 5 min window)
	reqTime, err := time.Parse(time.RFC3339, timestamp)
	if err != nil {
		return errors.New("invalid timestamp format")
	}

	if time.Since(reqTime) > 5*time.Minute || time.Since(reqTime) < -5*time.Minute {
		return errors.New("request expired or time skew too large")
	}

	// 2. Construct payload
	payload := fmt.Sprintf("%s%s%s%s", method, path, timestamp, body)

	// 3. Calculate HMAC
	h := hmac.New(sha256.New, []byte(secret))
	h.Write([]byte(payload))
	expectedSignature := hex.EncodeToString(h.Sum(nil))

	// 4. Verify signature
	if !hmac.Equal([]byte(signature), []byte(expectedSignature)) {
		return errors.New("invalid signature")
	}

	return nil
}

