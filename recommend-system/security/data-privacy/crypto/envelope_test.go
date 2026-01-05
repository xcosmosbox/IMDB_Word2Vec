package crypto

import (
	"testing"
)

func TestEnvelopeCrypto(t *testing.T) {
	kms := NewMockKMS()
	envelope := NewEnvelopeCrypto(kms)
	data := []byte("Sensitive Data")

	t.Run("EncryptDecrypt", func(t *testing.T) {
		encData, encKey, err := envelope.EncryptData(data)
		if err != nil {
			t.Fatalf("Envelope encrypt failed: %v", err)
		}

		if encData == "" || encKey == "" {
			t.Error("Empty encrypted output")
		}

		decrypted, err := envelope.DecryptData(encData, encKey)
		if err != nil {
			t.Fatalf("Envelope decrypt failed: %v", err)
		}

		if string(decrypted) != string(data) {
			t.Errorf("Expected %s, got %s", data, decrypted)
		}
	})
}

