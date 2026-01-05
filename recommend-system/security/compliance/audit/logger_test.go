package audit

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestAuditLogger(t *testing.T) {
	var buf bytes.Buffer
	logger := NewAuditLogger("test-service", &buf)

	err := logger.Log("test.event", "user-123", map[string]string{"foo": "bar"})
	if err != nil {
		t.Fatalf("Log failed: %v", err)
	}

	output := buf.String()
	if !strings.Contains(output, "test-service") {
		t.Error("Output missing service name")
	}
	if !strings.Contains(output, "user-123") {
		t.Error("Output missing user id")
	}

	var event AuditEvent
	if err := json.Unmarshal([]byte(output), &event); err != nil {
		t.Errorf("Invalid JSON output: %v", err)
	}
	
	if event.Type != "test.event" {
		t.Errorf("Expected event type test.event, got %s", event.Type)
	}
}

