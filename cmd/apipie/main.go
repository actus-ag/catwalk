// Package main provides a command-line tool to fetch models from APIpie
// and generate a configuration file for the provider.
//
// LLM-Enhanced Display Names:
// This tool uses APIpie.ai's LLM service to generate professional display names
// for models based on their IDs and descriptions. The API key is donated to
// improve the user experience of this open source project.
//
// API Key Configuration:
// Set APIPIE_DISPLAY_NAME_API_KEY environment variable to enable LLM-generated
// display names. This should be set in GitHub Actions secrets.
//
// Fallback Behavior:
// If the APIpie API key is not working or not provided, the tool will fall back
// to using the raw model ID as the display name. This ensures the tool never
// breaks due to API issues.
//
// GitHub Notification:
// If the APIpie API key fails, the tool will attempt to notify the configured
// GitHub user (set via APIPIE_API_KEY_NOTIFY_USER environment variable) about the issue.
//
// Example usage:
//
//	export APIPIE_DISPLAY_NAME_API_KEY="your-apipie-api-key"
//	export APIPIE_API_KEY_NOTIFY_USER="username-to-notify"
//	go run cmd/apipie/main.go
package main

import (
	"bytes"
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

// Model represents the complete model configuration from APIpie detailed endpoint.
type Model struct {
	ID                string   `json:"id"`
	Model             string   `json:"model"`
	Route             string   `json:"route,omitempty"`
	Description       string   `json:"description,omitempty"`
	MaxTokens         int64    `json:"max_tokens,omitempty"`
	MaxResponseTokens int64    `json:"max_response_tokens,omitempty"`
	Type              string   `json:"type,omitempty"`
	Subtype           string   `json:"subtype,omitempty"`
	Provider          string   `json:"provider,omitempty"`
	Enabled           int      `json:"enabled,omitempty"`
	Available         int      `json:"available,omitempty"`
	InputModalities   []string `json:"input_modalities,omitempty"`
	OutputModalities  []string `json:"output_modalities,omitempty"`
	Pricing           struct {
		Confirmed struct {
			InputCost  string `json:"input_cost"`
			OutputCost string `json:"output_cost"`
		} `json:"confirmed"`
		Advertised struct {
			InputCostPerToken  string `json:"input_cost_per_token"`
			OutputCostPerToken string `json:"output_cost_per_token"`
		} `json:"advertised"`
	} `json:"pricing"`
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
		"https://apipie.ai/v1/models/detailed",
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
	// Check if input modalities include image support
	return slices.Contains(model.InputModalities, "image") ||
		strings.Contains(model.Subtype, "multimodal") ||
		strings.Contains(model.Subtype, "vision") ||
		strings.Contains(strings.ToLower(model.Description), "vision") ||
		strings.Contains(strings.ToLower(model.Description), "image")
}

// APIpieRequest represents a request to the APIpie chat completions API
type APIpieRequest struct {
	Messages    []APIpieMessage `json:"messages"`
	Model       string          `json:"model"`
	MaxTokens   int             `json:"max_tokens"`
	Temperature float64         `json:"temperature"`
}

// APIpieMessage represents a message in the APIpie API request
type APIpieMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// APIpieResponse represents a response from the APIpie API
type APIpieResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// notifyGitHubUser attempts to log a message that can be picked up by GitHub Actions
// to notify a user about APIpie API key issues. This is a simple logging approach that
// GitHub Actions can parse and use to create issues or send notifications.
func notifyGitHubUser(message string) {
	if user := os.Getenv("APIPIE_API_KEY_NOTIFY_USER"); user != "" {
		// Log in a format that GitHub Actions can easily parse
		fmt.Printf("::warning title=APIpie API Key Issue::@%s %s\n", user, message)
		log.Printf("GitHub notification: @%s %s", user, message)
	}
}

// generateDisplayNameWithLLM uses APIpie.ai to generate professional display names
// for AI models based on their ID and description. This function is sponsored
// to improve the user experience of this open source project.
//
// Fallback: If the API key is not working or not provided, returns empty string
// and the caller should fall back to using the raw model ID as display name.
func generateDisplayNameWithLLM(id, description string) string {
	// Use dedicated API key for display name generation (donated for this project)
	apiKey := os.Getenv("APIPIE_DISPLAY_NAME_API_KEY")
	if apiKey == "" {
		return ""
	}

	// Create a comprehensive prompt for high-quality results
	prompt := fmt.Sprintf(`You are a model naming expert. Generate a clean, professional display name for an AI model.

Rules:
- Use proper capitalization (GPT-4, Claude 3.5, Llama 3.1, etc.)
- Keep version numbers and important identifiers
- Remove redundant words and technical jargon
- Make it user-friendly but informative
- Maximum 50 characters
- Follow established naming patterns from major providers

Examples:
- ID: "gpt-4o-2024-11-20" → "GPT-4o (2024-11-20)"
- ID: "claude-3-5-sonnet" → "Claude 3.5 Sonnet"
- ID: "llama-3-1-70b-instruct" → "Llama 3.1 70B Instruct"
- ID: "mistral-7b-instruct-v0-3" → "Mistral 7B Instruct v0.3"

Model ID: "%s"
Description: "%s"

Generate only the display name, nothing else:`, id, strings.Split(description, "\n")[0])

	reqBody := APIpieRequest{
		Messages: []APIpieMessage{
			{
				Role:    "user",
				Content: prompt,
			},
		},
		Model:       "claude-sonnet-4",
		MaxTokens:   100,
		Temperature: 0.1, // Low temperature for consistent results
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		notifyGitHubUser("Failed to marshal APIpie request for display name generation")
		return ""
	}

	client := &http.Client{Timeout: 30 * time.Second}
	req, err := http.NewRequestWithContext(
		context.Background(),
		"POST",
		"https://apipie.ai/v1/chat/completions",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		notifyGitHubUser("Failed to create APIpie request for display name generation")
		return ""
	}

	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		notifyGitHubUser("APIpie API request failed for display name generation - network error")
		return ""
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		notifyGitHubUser(fmt.Sprintf("APIpie API returned status %d for display name generation: %s", resp.StatusCode, string(body)))
		return ""
	}

	var apipieResp APIpieResponse
	if err := json.NewDecoder(resp.Body).Decode(&apipieResp); err != nil {
		notifyGitHubUser("Failed to decode APIpie response for display name generation")
		return ""
	}

	if len(apipieResp.Choices) == 0 {
		notifyGitHubUser("APIpie returned empty choices for display name generation")
		return ""
	}

	// Clean up the response
	name := strings.TrimSpace(apipieResp.Choices[0].Message.Content)
	name = strings.Trim(name, "\"'") // Remove quotes if present

	// Validate the response (basic sanity check)
	if len(name) > 0 && len(name) <= 60 && !strings.Contains(name, "\n") {
		return name
	}

	notifyGitHubUser(fmt.Sprintf("APIpie returned invalid display name format: '%s'", name))
	return ""
}

// createDisplayName generates a display name for a model using LLM when possible.
// If the APIpie API key is not working or not provided, falls back to using
// the raw model ID as the display name to ensure the tool never breaks.
func createDisplayName(model Model) string {
	// Try LLM generation first (if API key is available and working)
	if llmName := generateDisplayNameWithLLM(model.ID, model.Description); llmName != "" {
		return llmName
	}

	// Fallback: Use the raw model ID as display name
	// This ensures the tool never breaks due to API issues
	return model.ID
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
		DefaultLargeModelID: "claude-sonnet-4",
		DefaultSmallModelID: "claude-3-5-haiku",
		Models:              []catwalk.Model{},
	}

	for _, model := range modelsResp.Data {
		// Skip non-text models
		if !isTextModel(model) {
			continue
		}

		// Parse and convert costs from per-token to per-million-tokens
		// Try confirmed pricing first, fall back to advertised
		var inputCostPerToken, outputCostPerToken float64

		if model.Pricing.Confirmed.InputCost != "" {
			inputCostPerToken, _ = strconv.ParseFloat(model.Pricing.Confirmed.InputCost, 64)
		} else if model.Pricing.Advertised.InputCostPerToken != "" {
			inputCostPerToken, _ = strconv.ParseFloat(model.Pricing.Advertised.InputCostPerToken, 64)
		}

		if model.Pricing.Confirmed.OutputCost != "" {
			outputCostPerToken, _ = strconv.ParseFloat(model.Pricing.Confirmed.OutputCost, 64)
		} else if model.Pricing.Advertised.OutputCostPerToken != "" {
			outputCostPerToken, _ = strconv.ParseFloat(model.Pricing.Advertised.OutputCostPerToken, 64)
		}

		costPer1MIn := inputCostPerToken * 1_000_000
		costPer1MOut := outputCostPerToken * 1_000_000

		m := catwalk.Model{
			ID:                 model.ID,
			Name:               createDisplayName(model),
			CostPer1MIn:        costPer1MIn,
			CostPer1MOut:       costPer1MOut,
			CostPer1MInCached:  costPer1MIn * 0.5,   // Assume 50% discount for cached
			CostPer1MOutCached: costPer1MOut * 0.25, // Assume 75% discount for cached output
			ContextWindow:      getContextWindow(model),
			DefaultMaxTokens:   getDefaultMaxTokens(model),
			CanReason:          false, // APIpie doesn't specify reasoning capabilities
			HasReasoningEffort: false,
			SupportsImages:     supportsImages(model),
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
