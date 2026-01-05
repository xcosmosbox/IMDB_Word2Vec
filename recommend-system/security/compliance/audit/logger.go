package audit

import (
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/google/uuid"
)

// AuditEvent follows CloudEvents spec
type AuditEvent struct {
	ID          string      `json:"id"`
	Source      string      `json:"source"`
	SpecVersion string      `json:"specversion"`
	Type        string      `json:"type"` // e.g., "auth.login.success"
	Time        time.Time   `json:"time"`
	Subject     string      `json:"subject"` // User ID
	Data        interface{} `json:"data"`
}

type AuditLogger struct {
	ServiceName string
	Output      io.Writer
}

func NewAuditLogger(serviceName string, output io.Writer) *AuditLogger {
	return &AuditLogger{
		ServiceName: serviceName,
		Output:      output,
	}
}

func (l *AuditLogger) Log(eventType string, userID string, details interface{}) error {
	event := AuditEvent{
		ID:          uuid.New().String(),
		Source:      l.ServiceName,
		SpecVersion: "1.0",
		Type:        eventType,
		Time:        time.Now(),
		Subject:     userID,
		Data:        details,
	}

	bytes, err := json.Marshal(event)
	if err != nil {
		return err
	}

	_, err = fmt.Fprintln(l.Output, string(bytes))
	return err
}

