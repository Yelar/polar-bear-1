//
//  ContentView.swift
//  polar-bear
//
//  Created by Yelarys Yertaiuly on 17/01/2026.
//

import SwiftUI

struct SettingsView: View {
    @AppStorage("tokenc.backendUrl") private var backendUrl = "http://127.0.0.1:8000/compress"
    @AppStorage("tokenc.aggressiveness") private var aggressiveness = 0.5

    var body: some View {
        Form {
            Section("Backend") {
                TextField("Backend URL", text: $backendUrl)
                    .textContentType(.URL)
                    .autocorrectionDisabled()
                HStack {
                    Text("Aggressiveness")
                    Slider(value: $aggressiveness, in: 0...1, step: 0.05)
                    Text(String(format: "%.2f", aggressiveness))
                        .frame(width: 48, alignment: .trailing)
                }
                Text("Run: uvicorn main:app --reload")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            Section("Hot Key") {
                Text("Press Command + Option to compress the focused text field.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            Section("Notes") {
                Text("Accessibility permission is required to read and replace text in other apps.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .frame(width: 420)
    }
}

#Preview {
    SettingsView()
}
