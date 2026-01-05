package mfa

import (
	"crypto/hmac"
	"crypto/sha1"
	"encoding/base32"
	"encoding/binary"
	"fmt"
	"time"
)

// TOTPService handles Time-based One-Time Password generation and validation
type TOTPService struct {
	Issuer string
}

func NewTOTPService(issuer string) *TOTPService {
	return &TOTPService{Issuer: issuer}
}

// GenerateSecret generates a random secret key for TOTP
func (s *TOTPService) GenerateSecret() (string, error) {
	// In a real implementation, generate random bytes
	// Here we return a fixed string for demonstration or use a simple generator
	// Ideally use crypto/rand
	secret := []byte("JBSWY3DPEHPK3PXP") // Example base32
	return string(secret), nil
}

// ValidateCode validates a TOTP code against a secret
func (s *TOTPService) ValidateCode(secret, code string) bool {
	// Normalize secret
	secretKey, err := base32.StdEncoding.DecodeString(secret)
	if err != nil {
		return false
	}

	// Calculate current time step (30 seconds)
	epoch := time.Now().Unix()
	timeStep := 30
	counter := epoch / int64(timeStep)

	// Check current and adjacent windows for clock skew
	for i := -1; i <= 1; i++ {
		hash := hmacSha1(secretKey, counter+int64(i))
		offset := hash[len(hash)-1] & 0x0F
		binaryCode := (int(hash[offset]&0x7F) << 24) |
			(int(hash[offset+1]&0xFF) << 16) |
			(int(hash[offset+2]&0xFF) << 8) |
			(int(hash[offset+3]&0xFF))
		otp := binaryCode % 1000000
		
		if fmt.Sprintf("%06d", otp) == code {
			return true
		}
	}

	return false
}

func hmacSha1(key []byte, counter int64) []byte {
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, uint64(counter))
	h := hmac.New(sha1.New, key)
	h.Write(buf)
	return h.Sum(nil)
}

