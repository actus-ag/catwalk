// Package main provides a command-line tool to fetch models from APIpie
// and generate a configuration file for the provider.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/catwalk/pkg/catwalk"
)

// Model represents the complete model configuration from APIpie.
type Model struct {
	ID                  string `json:"id"`
	Model               string `json:"model"`
	Route               string `json:"route,omitempty"`
	Description         string `json:"description,omitempty"`
	MaxTokens           int64  `json:"max_tokens,omitempty"`
	MaxResponseTokens   int64  `json:"max_response_tokens,omitempty"`
	InputCost           string `json:"input_cost,omitempty"`
	OutputCost          string `json:"output_cost,omitempty"`
	Type                string `json:"type,omitempty"`
	Subtype             string `json:"subtype,omitempty"`
	Provider            string `json:"provider,omitempty"`
	Enabled             int    `json:"enabled,omitempty"`
	Available           int    `json:"available,omitempty"`
}

// ModelsResponse is the response structure for the models API.
type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

func fetchAPIpieModels() (*ModelsResponse, error) {
	client := &http.Client{Timeout: 30 * time.Second}
	req, _ := http.NewRequestWithContext(
		context.Background(),
		"GET",
		"https://apipie.ai/v1/models",
		nil,
	)
	req.Header.Set("User-Agent", "Catwalk-Client/1.0")
	
	// Try to use API key if available
	if apiKey := os.Getenv("APIPIE_API_KEY"); apiKey != "" {
		req.Header.Set("x-api-key", apiKey)
	}
	
	resp, err := client.Do(req)
	if err != nil {
		return nil, err //nolint:wrapcheck
	}
	defer resp.Body.Close() //nolint:errcheck
	
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, body)
	}
	
	var mr ModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&mr); err != nil {
		return nil, err //nolint:wrapcheck
	}
	return &mr, nil
}

func isTextModel(model Model) bool {
	// Check if model is enabled, available, and is an LLM
	return model.Enabled == 1 && model.Available == 1 && model.Type == "llm"
}

func supportsImages(model Model) bool {
	// Check if subtype includes vision capabilities
	return strings.Contains(model.Subtype, "vision") || 
		   strings.Contains(strings.ToLower(model.Description), "vision") ||
		   strings.Contains(strings.ToLower(model.Description), "image")
}

func getDefaultMaxTokens(model Model) int64 {
	if model.MaxResponseTokens > 0 {
		return model.MaxResponseTokens
	}
	if model.MaxTokens > 0 {
		return model.MaxTokens / 4 // Conservative default
	}
	return 4096 // reasonable default
}

func getContextWindow(model Model) int64 {
	if model.MaxTokens > 0 {
		return model.MaxTokens
	}
	// Fallback defaults based on common model patterns
	switch {
	case strings.Contains(strings.ToLower(model.ID), "gpt-4o"):
		return 128000
	case strings.Contains(strings.ToLower(model.ID), "gpt-4"):
		return 8192
	case strings.Contains(strings.ToLower(model.ID), "gpt-3.5"):
		return 16385
	case strings.Contains(strings.ToLower(model.ID), "claude"):
		return 200000
	case strings.Contains(strings.ToLower(model.ID), "gemini"):
		return 32768
	case strings.Contains(strings.ToLower(model.ID), "llama"):
		return 128000
	default:
		return 32768
	}
}

// This is used to generate the apipie.json config file.
func main() {
	modelsResp, err := fetchAPIpieModels()
	if err != nil {
		log.Fatal("Error fetching APIpie models:", err)
	}

	apipieProvider := catwalk.Provider{
		Name:                "APIpie",
		ID:                  "apipie",
		APIKey:              "$APIPIE_API_KEY",
		APIEndpoint:         "https://apipie.ai/v1",
		Type:                catwalk.TypeOpenAI,
		DefaultLargeModelID: "gpt-4o",
		DefaultSmallModelID: "gpt-4o-mini",
		Models:              []catwalk.Model{},
	}

	for _, model := range modelsResp.Data {
		// Skip non-text models
		if !isTextModel(model) {
			continue
		}

		// Parse and convert costs from per-token to per-million-tokens
		inputCost, err := strconv.ParseFloat(model.InputCost, 64)
		if err != nil {
			inputCost = 0.0
		}
		outputCost, err := strconv.ParseFloat(model.OutputCost, 64)
		if err != nil {
			outputCost = 0.0
		}
		
		costPer1MIn := inputCost * 1_000_000
		costPer1MOut := outputCost * 1_000_000

		m := catwalk.Model{
			ID:                 model.ID,
			Name:               model.ID, // Use ID as name if no description
			CostPer1MIn:        costPer1MIn,
			CostPer1MOut:       costPer1MOut,
			CostPer1MInCached:  costPer1MIn * 0.5, // Assume 50% discount for cached
			CostPer1MOutCached: costPer1MOut * 0.25, // Assume 75% discount for cached output
			ContextWindow:      getContextWindow(model),
			DefaultMaxTokens:   getDefaultMaxTokens(model),
			CanReason:          false, // APIpie doesn't specify reasoning capabilities
			HasReasoningEffort: false,
			SupportsImages:     supportsImages(model),
		}

		// Use description as name if available
		if model.Description != "" {
			m.Name = model.Description
		}

		apipieProvider.Models = append(apipieProvider.Models, m)
		fmt.Printf("Added model %s with context window %d\n", model.ID, m.ContextWindow)
	}

	// Sort models by name for consistency
	slices.SortFunc(apipieProvider.Models, func(a catwalk.Model, b catwalk.Model) int {
		return strings.Compare(a.Name, b.Name)
	})

	// Save the JSON in internal/providers/configs/apipie.json
	data, err := json.MarshalIndent(apipieProvider, "", "  ")
	if err != nil {
		log.Fatal("Error marshaling APIpie provider:", err)
	}

	// Write to file
	if err := os.WriteFile("internal/providers/configs/apipie.json", data, 0o600); err != nil {
		log.Fatal("Error writing APIpie provider config:", err)
	}

	fmt.Printf("Successfully generated APIpie provider config with %d models\n", len(apipieProvider.Models))
}