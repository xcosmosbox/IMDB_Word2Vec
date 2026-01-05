package crypto

import (
	"crypto/rand"
	"encoding/base64"
	"errors"
)

// KMSClient interface for dependency injection
type KMSClient interface {
	Encrypt(plaintext []byte) ([]byte, error)
	Decrypt(ciphertext []byte) ([]byte, error)
}

// MockKMS is a simple mock for testing or local development
type MockKMS struct {
	MasterKey []byte
}

func NewMockKMS() *MockKMS {
	key := make([]byte, 32)
	rand.Read(key)
	return &MockKMS{MasterKey: key}
}

func (k *MockKMS) Encrypt(plaintext []byte) ([]byte, error) {
	// Simple XOR for mock purposes or just wrapper
	// In reality, this calls AWS KMS / HashiCorp Vault
	// Here we return base64 encoded plaintext prefixed with "kms:" for simulation
	return []byte("kms:" + base64.StdEncoding.EncodeToString(plaintext)), nil
}

func (k *MockKMS) Decrypt(ciphertext []byte) ([]byte, error) {
	s := string(ciphertext)
	if len(s) < 4 || s[:4] != "kms:" {
		return nil, errors.New("invalid kms ciphertext")
	}
	return base64.StdEncoding.DecodeString(s[4:])
}

// EnvelopeCrypto handles envelope encryption
type EnvelopeCrypto struct {
	kms KMSClient
}

func NewEnvelopeCrypto(kms KMSClient) *EnvelopeCrypto {
	return &EnvelopeCrypto{kms: kms}
}

// EncryptData generates a DEK, encrypts data with DEK, encrypts DEK with KMS
func (e *EnvelopeCrypto) EncryptData(data []byte) (string, string, error) {
	// 1. Generate Data Key (DEK)
	dek := make([]byte, 32)
	if _, err := rand.Read(dek); err != nil {
		return "", "", err
	}

	// 2. Encrypt Data with DEK
	dekBase64 := base64.StdEncoding.EncodeToString(dek)
	localCrypto, err := NewCryptoService(dekBase64)
	if err != nil {
		return "", "", err
	}
	encryptedData, err := localCrypto.Encrypt(data)
	if err != nil {
		return "", "", err
	}

	// 3. Encrypt DEK with KMS
	encryptedDEK, err := e.kms.Encrypt(dek)
	if err != nil {
		return "", "", err
	}

	return encryptedData, base64.StdEncoding.EncodeToString(encryptedDEK), nil
}

// DecryptData decrypts DEK with KMS, then decrypts data with DEK
func (e *EnvelopeCrypto) DecryptData(encryptedData string, encryptedDEKBase64 string) ([]byte, error) {
	// 1. Decrypt DEK
	encryptedDEK, err := base64.StdEncoding.DecodeString(encryptedDEKBase64)
	if err != nil {
		return nil, err
	}
	dek, err := e.kms.Decrypt(encryptedDEK)
	if err != nil {
		return nil, err
	}

	// 2. Decrypt Data
	dekBase64 := base64.StdEncoding.EncodeToString(dek)
	localCrypto, err := NewCryptoService(dekBase64)
	if err != nil {
		return nil, err
	}
	
	return localCrypto.Decrypt(encryptedData)
}

