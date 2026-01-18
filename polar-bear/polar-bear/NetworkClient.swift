//
//  NetworkClient.swift
//  polar-bear
//
//  Created by Codex.
//

import Foundation

// MARK: - Compression Result

struct CompressionResult {
    let output: String
    let originalTokens: Int
    let compressedTokens: Int
    let tokensSaved: Int
    
    var reductionRatio: Double {
        guard originalTokens > 0 else { return 0 }
        return Double(tokensSaved) / Double(originalTokens)
    }
}

// MARK: - Network Client

struct NetworkClient {
    let aggressiveness: Double
    let backendUrl: String
    let provider: String
    let tokencApiKey: String?
    
    /// Threshold for auto-routing: prompts below this use TokenC, above use local
    static let autoRoutingThreshold = 500

    init(
        backendUrl: String,
        aggressiveness: Double = 0.5,
        provider: String = "local",
        tokencApiKey: String? = nil
    ) {
        self.aggressiveness = aggressiveness
        self.backendUrl = backendUrl
        self.provider = provider
        self.tokencApiKey = tokencApiKey
    }
    
    /// Determine the provider and mode based on text length and settings
    /// All requests go through the local backend - backend handles TokenC routing
    private func resolveProvider(for text: String) -> (provider: String, mode: String) {
        let estimatedTokens = estimateTokens(text)
        
        // Auto mode: route based on token count
        if provider == "auto" {
            if estimatedTokens < Self.autoRoutingThreshold {
                // Short prompts -> TokenC via backend
                return (provider: "tokenc", mode: "ml")
            } else {
                // Long prompts -> Local LLMLingua
                return (provider: "local", mode: "ml")
            }
        }
        
        // Manual provider selection
        switch provider {
        case "tokenc":
            return (provider: "tokenc", mode: "ml")
        case "hybrid":
            return (provider: "local", mode: "hybrid")
        default: // "local" or "ml"
            return (provider: "local", mode: "ml")
        }
    }
    
    /// Get the backend endpoint URL
    private var endpoint: URL {
        URL(string: backendUrl) ?? URL(string: "http://127.0.0.1:8000/compress")!
    }

    /// Compress text and return full result with stats
    func compressText(_ input: String) async throws -> CompressionResult {
        // Resolve provider based on text length and settings
        let resolved = resolveProvider(for: input)
        let estimatedTokens = estimateTokens(input)
        
        // ALL requests go through local backend - backend handles TokenC routing
        print("[NetworkClient] Routing: \(estimatedTokens) tokens -> \(resolved.provider) via backend @ \(endpoint)")
        
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 60 // Longer timeout for LLMLingua

        let body = CompressRequest(
            input: input,
            aggressiveness: aggressiveness,
            mode: resolved.mode,
            provider: resolved.provider
        )
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            print("[NetworkClient] Error: Invalid response")
            throw NetworkError.invalidResponse
        }
        
        if !(200..<300).contains(httpResponse.statusCode) {
            let payload = String(data: data, encoding: .utf8) ?? ""
            print("[NetworkClient] Error: HTTP \(httpResponse.statusCode) - \(payload.prefix(200))")
            throw NetworkError.httpStatus(httpResponse.statusCode, payload)
        }

        let decoded = try JSONDecoder().decode(CompressResponse.self, from: data)
        
        // Use backend stats if available, otherwise estimate
        let originalTokens = decoded.originalTokens ?? estimateTokens(input)
        let compressedTokens = decoded.compressedTokens ?? estimateTokens(decoded.output)
        let tokensSaved = decoded.tokensSaved ?? max(0, originalTokens - compressedTokens)
        
        print("[NetworkClient] Success: \(originalTokens) -> \(compressedTokens) tokens")
        
        return CompressionResult(
            output: decoded.output,
            originalTokens: originalTokens,
            compressedTokens: compressedTokens,
            tokensSaved: tokensSaved
        )
    }
    
    /// Simple token estimation (words * 1.3)
    private func estimateTokens(_ text: String) -> Int {
        let words = text.split(separator: " ").count
        return Int(Double(words) * 1.3)
    }
}

// MARK: - Network Error

enum NetworkError: LocalizedError {
    case invalidResponse
    case httpStatus(Int, String)
    
    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid response from server"
        case .httpStatus(let code, let message):
            return "HTTP \(code): \(message)"
        }
    }
}

// MARK: - Request/Response Models

private struct CompressRequest: Encodable {
    let input: String
    let aggressiveness: Double
    let mode: String
    let provider: String
}

private struct CompressResponse: Decodable {
    let output: String
    let originalTokens: Int?
    let compressedTokens: Int?
    let tokensSaved: Int?
    let reductionRatio: Double?
    let method: String?
    
    enum CodingKeys: String, CodingKey {
        case output
        case originalTokens = "original_tokens"
        case compressedTokens = "compressed_tokens"
        case tokensSaved = "tokens_saved"
        case reductionRatio = "reduction_ratio"
        case method
    }
}
