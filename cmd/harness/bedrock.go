package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	brdoc "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

// chatStreamBedrock uses the AWS SDK's ConverseStream API for Bedrock.
func (sc *StreamingClient) chatStreamBedrock(messages []map[string]interface{}, onChunk func(StreamChunk)) error {
	// Extract region from API URL (e.g. "https://bedrock-runtime.us-east-1.amazonaws.com")
	region := extractBedrockRegion(sc.APIBase)

	// Set bearer token env var for AWS SDK credential chain
	os.Setenv("AWS_BEARER_TOKEN_BEDROCK", sc.APIKey)

	// Create AWS config with static dummy credentials (Bedrock API key auth
	// doesn't use standard SigV4, but the SDK requires credentials to be set)
	cfg, err := awsconfig.LoadDefaultConfig(context.Background(),
		awsconfig.WithRegion(region),
		awsconfig.WithCredentialsProvider(
			credentials.NewStaticCredentialsProvider("BEDROCK", sc.APIKey, ""),
		),
	)
	if err != nil {
		return fmt.Errorf("bedrock: failed to load AWS config: %w", err)
	}

	client := bedrockruntime.NewFromConfig(cfg)

	// Convert messages to Bedrock format
	bedrockMessages, systemContent := convertToBedrock(messages)

	// Build request
	input := &bedrockruntime.ConverseStreamInput{
		ModelId:  aws.String(sc.Model),
		Messages: bedrockMessages,
		InferenceConfig: &types.InferenceConfiguration{
			Temperature: aws.Float32(float32(sc.Temperature)),
			MaxTokens:   aws.Int32(int32(sc.MaxTokens)),
		},
	}
	if systemContent != nil {
		input.System = systemContent
	}

	// Enable thinking/reasoning for models that support it.
	// Claude uses {"thinking": {"type": "enabled", "budget_tokens": N}}
	// Qwen3 uses {"enable_thinking": true}
	if sc.ReasoningEffort != "none" {
		budget := sc.MaxTokens / 2
		if budget < 1024 {
			budget = 1024
		}
		if strings.Contains(strings.ToLower(sc.Model), "qwen") {
			input.AdditionalModelRequestFields = brdoc.NewLazyDocument(map[string]interface{}{
				"enable_thinking": true,
			})
		} else {
			input.AdditionalModelRequestFields = brdoc.NewLazyDocument(map[string]interface{}{
				"thinking": map[string]interface{}{
					"type":          "enabled",
					"budget_tokens": budget,
				},
			})
		}
	}

	// Call ConverseStream
	output, err := client.ConverseStream(context.Background(), input)
	if err != nil {
		errMsg := strings.ToLower(err.Error())

		// If thinking is unsupported by this model, retry without it
		if input.AdditionalModelRequestFields != nil &&
			(strings.Contains(errMsg, "thinking") || strings.Contains(errMsg, "additional")) {
			input.AdditionalModelRequestFields = nil
			output, err = client.ConverseStream(context.Background(), input)
		}

		// Auto-retry with us. prefix if bare model ID is rejected
		if err != nil {
			if strings.Contains(errMsg, "inference profile") ||
				strings.Contains(errMsg, "validation") {
				prefixed := "us." + sc.Model
				input.ModelId = aws.String(prefixed)
				output, err = client.ConverseStream(context.Background(), input)
				if err != nil {
					return fmt.Errorf("bedrock: %w", err)
				}
				sc.Model = prefixed // cache for future calls
			} else {
				return fmt.Errorf("bedrock: %w", err)
			}
		}
	}

	// Process event stream
	stream := output.GetStream()
	defer stream.Close()

	for event := range stream.Events() {
		if isInterrupted() {
			onChunk(StreamChunk{FinishReason: "interrupted"})
			break
		}

		switch ev := event.(type) {
		case *types.ConverseStreamOutputMemberContentBlockDelta:
			delta := ev.Value.Delta
			switch d := delta.(type) {
			case *types.ContentBlockDeltaMemberText:
				if d.Value != "" {
					onChunk(StreamChunk{Content: d.Value})
				}
			case *types.ContentBlockDeltaMemberReasoningContent:
				rc := d.Value
				switch r := rc.(type) {
				case *types.ReasoningContentBlockDeltaMemberText:
					if r.Value != "" {
						onChunk(StreamChunk{Reasoning: r.Value})
					}
				}
			}

		case *types.ConverseStreamOutputMemberMessageStop:
			reason := string(ev.Value.StopReason)
			if reason != "" {
				onChunk(StreamChunk{FinishReason: strings.ToLower(reason)})
			}

		case *types.ConverseStreamOutputMemberMetadata:
			// Usage info available here if needed
		}
	}

	if err := stream.Err(); err != nil {
		return fmt.Errorf("bedrock stream error: %w", err)
	}

	return nil
}

// convertToBedrock converts OpenAI-format messages to Bedrock Converse format.
func convertToBedrock(messages []map[string]interface{}) ([]types.Message, []types.SystemContentBlock) {
	var bedrockMsgs []types.Message
	var system []types.SystemContentBlock

	for _, m := range messages {
		role, _ := m["role"].(string)
		content, _ := m["content"].(string)

		if role == "system" {
			system = append(system, &types.SystemContentBlockMemberText{Value: content})
			continue
		}

		// Map "assistant" to "assistant", "user" to "user"
		brRole := types.ConversationRoleUser
		if role == "assistant" {
			brRole = types.ConversationRoleAssistant
		}

		bedrockMsgs = append(bedrockMsgs, types.Message{
			Role:    brRole,
			Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: content}},
		})
	}

	return bedrockMsgs, system
}

// extractBedrockRegion extracts the AWS region from a Bedrock URL.
func extractBedrockRegion(url string) string {
	// https://bedrock-runtime.us-east-1.amazonaws.com
	url = strings.TrimPrefix(url, "https://")
	url = strings.TrimPrefix(url, "http://")
	parts := strings.Split(url, ".")
	// bedrock-runtime.REGION.amazonaws.com
	if len(parts) >= 3 {
		return parts[1]
	}
	return "us-east-1" // default
}
