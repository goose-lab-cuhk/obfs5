// Package config holds the endpoint's settings (config.yaml). The file is built
// into the binary; Load applies the changes of a copy given with -config.
package config

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// Config mirrors config.yaml.
type Config struct {
	Client    Client    `yaml:"client"`
	Server    Server    `yaml:"server"`
	Tunnel    Tunnel    `yaml:"tunnel"`
	Scheduler Scheduler `yaml:"scheduler"`
	KCP       KCP       `yaml:"kcp"`
}

type Client struct {
	Mode     string  `yaml:"mode"`
	Conns    int     `yaml:"conns"`
	Watchdog Seconds `yaml:"watchdog"`
}

type Server struct {
	SessionIdle Seconds `yaml:"session_idle"`
}

type Tunnel struct {
	MaxLife          Seconds `yaml:"max_life"`
	HandshakeTimeout Seconds `yaml:"handshake_timeout"`
}

// Scheduler: the asynchronous mode's deadline for a predicted receive,
// last event + Alpha·quantile(λ) + RTT + Beta, with λ uniform in Lambda.
type Scheduler struct {
	Alpha     float64    `yaml:"alpha"`
	Beta      float64    `yaml:"beta"`
	Lambda    [2]float64 `yaml:"lambda"`
	MaxRounds int        `yaml:"max_rounds"`
}

type KCP struct {
	Window int `yaml:"window"`
	MTU    int `yaml:"mtu"`
}

// Seconds is a time written in seconds.
type Seconds float64

func (s Seconds) Duration() time.Duration { return time.Duration(float64(s) * float64(time.Second)) }

// Load returns the built-in settings with the changes of the file at path
// (none when path is empty). An unknown key is an error.
func Load(builtin []byte, path string) (*Config, error) {
	var c Config
	if err := decode(builtin, &c); err != nil {
		return nil, fmt.Errorf("built-in config.yaml: %w", err)
	}
	if path != "" {
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		if err := decode(b, &c); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
	}
	if c.Client.Mode != "async" && c.Client.Mode != "sync" {
		return nil, errors.New("config: client.mode must be async or sync")
	}
	return &c, nil
}

func decode(b []byte, c *Config) error {
	d := yaml.NewDecoder(bytes.NewReader(b))
	d.KnownFields(true)
	if err := d.Decode(c); err != nil && err != io.EOF { // io.EOF: an empty file
		return err
	}
	return nil
}
