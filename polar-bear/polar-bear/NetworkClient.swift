//
//  NetworkClient.swift
//  polar-bear
//
//  Created by Codex.
//

import Foundation

struct NetworkClient {
    let aggressiveness: Double
    let endpoint: URL

    init(backendUrl: String, aggressiveness: Double = 0.5) {
        self.aggressiveness = aggressiveness
        self.endpoint = URL(string: backendUrl) ?? URL(string: "http://127.0.0.1:8000/compress")!
    }

    func compressText(_ input: String) async throws -> String {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = CompressRequest(input: input, aggressiveness: aggressiveness)
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let payload = String(data: data, encoding: .utf8) ?? ""
            throw NetworkError.httpStatus(httpResponse.statusCode, payload)
        }

        let decoded = try JSONDecoder().decode(CompressResponse.self, from: data)
        return decoded.output
    }
}

enum NetworkError: Error {
    case invalidResponse
    case httpStatus(Int, String)
}

private struct CompressRequest: Encodable {
    let input: String
    let aggressiveness: Double
}

private struct CompressResponse: Decodable {
    let output: String
}
