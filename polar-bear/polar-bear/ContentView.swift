//
//  ContentView.swift
//  polar-bear
//
//  Created by Yelarys Yertaiuly on 17/01/2026.
//

import SwiftUI

// MARK: - Settings View

struct SettingsView: View {
    @AppStorage("tokenc.backendUrl") private var backendUrl = "http://127.0.0.1:8000/compress"
    @AppStorage("tokenc.aggressiveness") private var aggressiveness = 0.5
    @AppStorage("tokenc.provider") private var provider = "auto"
    @AppStorage("tokenc.totalTokensSaved") private var totalTokensSaved = 0
    @AppStorage("tokenc.totalCompressions") private var totalCompressions = 0
    @AppStorage("tokenc.tokencApiKey") private var tokencApiKey = ""
    
    @State private var selectedTab = 0
    
    var body: some View {
        VStack(spacing: 0) {
            // Custom Tab Bar
            HStack(spacing: 0) {
                TabButton(title: "Statistics", icon: "chart.bar.fill", isSelected: selectedTab == 0) {
                    selectedTab = 0
                }
                TabButton(title: "Settings", icon: "gear", isSelected: selectedTab == 1) {
                    selectedTab = 1
                }
                TabButton(title: "About", icon: "info.circle.fill", isSelected: selectedTab == 2) {
                    selectedTab = 2
                }
            }
            .padding(.horizontal, 12)
            .padding(.top, 8)
            
            Divider()
                .padding(.top, 8)
            
            // Tab Content
            Group {
                switch selectedTab {
                case 0:
                    StatsTabView(
                        totalTokensSaved: totalTokensSaved,
                        totalCompressions: totalCompressions,
                        onReset: resetStats
                    )
                case 1:
                    ProviderTabView(
                        provider: $provider,
                        tokencApiKey: $tokencApiKey,
                        backendUrl: $backendUrl,
                        aggressiveness: $aggressiveness
                    )
                case 2:
                    AboutTabView()
                default:
                    EmptyView()
                }
            }
        }
        .frame(width: 500, height: 520)
    }
    
    private func resetStats() {
        totalTokensSaved = 0
        totalCompressions = 0
    }
}

// MARK: - Tab Button

struct TabButton: View {
    let title: String
    let icon: String
    let isSelected: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 18))
                Text(title)
                    .font(.caption)
            }
            .foregroundStyle(isSelected ? .blue : .secondary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(isSelected ? Color.blue.opacity(0.1) : Color.clear)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Stats Tab

struct StatsTabView: View {
    let totalTokensSaved: Int
    let totalCompressions: Int
    let onReset: () -> Void
    
    private var estimatedCostSaved: Double {
        // Claude Opus 4.5 pricing: $15 per 1M input tokens
        Double(totalTokensSaved) * 0.000015
    }
    
    private var averageTokensPerCompression: Int {
        guard totalCompressions > 0 else { return 0 }
        return totalTokensSaved / totalCompressions
    }

    var body: some View {
        VStack(spacing: 24) {
            // Header with polar bear icon
            VStack(spacing: 8) {
                Image(systemName: "snowflake")
                    .font(.system(size: 48))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.cyan, .blue],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .shadow(color: .cyan.opacity(0.5), radius: 10)
                
                Text("Token Savings")
                    .font(.title2.bold())
                    .foregroundStyle(.primary)
            }
            .padding(.top, 8)
            
            // Stats Cards
            HStack(spacing: 16) {
                StatCard(
                    title: "Tokens Saved",
                    value: formatNumber(totalTokensSaved),
                    icon: "arrow.down.circle.fill",
                    color: .green
                )
                
                StatCard(
                    title: "Compressions",
                    value: "\(totalCompressions)",
                    icon: "repeat.circle.fill",
                    color: .blue
                )
            }
            
            HStack(spacing: 16) {
                StatCard(
                    title: "Avg. Saved",
                    value: formatNumber(averageTokensPerCompression),
                    icon: "chart.line.downtrend.xyaxis.circle.fill",
                    color: .orange
                )
                
                StatCard(
                    title: "Cost Saved (Opus 4.5)",
                    value: formatCost(estimatedCostSaved),
                    icon: "dollarsign.circle.fill",
                    color: .purple
                )
            }
            
            Spacer()
            
            // Reset button
            Button(action: onReset) {
                Label("Reset Statistics", systemImage: "arrow.counterclockwise")
                    .font(.callout)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .padding(.bottom, 8)
        }
        .padding(20)
    }
    
    private func formatNumber(_ num: Int) -> String {
        if num >= 1_000_000 {
            return String(format: "%.1fM", Double(num) / 1_000_000)
        } else if num >= 1_000 {
            return String(format: "%.1fK", Double(num) / 1_000)
        }
        return "\(num)"
    }
    
    private func formatCost(_ cost: Double) -> String {
        if cost >= 1.0 {
            return String(format: "$%.2f", cost)
        } else if cost >= 0.01 {
            return String(format: "$%.3f", cost)
        } else {
            return String(format: "$%.4f", cost)
        }
    }
}

struct StatCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 28))
                .foregroundStyle(color)
            
            Text(value)
                .font(.system(size: 24, weight: .bold, design: .rounded))
                .foregroundStyle(.primary)
            
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(color.opacity(0.1))
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .strokeBorder(color.opacity(0.2), lineWidth: 1)
                )
        )
    }
}

// MARK: - Provider Tab

struct ProviderTabView: View {
    @Binding var provider: String
    @Binding var tokencApiKey: String
    @Binding var backendUrl: String
    @Binding var aggressiveness: Double
    
    private var compressionLabel: String {
        switch aggressiveness {
        case 0..<0.3: return "Light"
        case 0.3..<0.6: return "Medium"
        case 0.6..<0.8: return "Strong"
        default: return "Maximum"
        }
    }
    
    private var compressionColor: Color {
        switch aggressiveness {
        case 0..<0.3: return .green
        case 0.3..<0.6: return .blue
        case 0.6..<0.8: return .orange
        default: return .red
        }
    }

    var body: some View {
        ScrollView {
        VStack(alignment: .leading, spacing: 20) {
            // COMPRESSION SLIDER - Most important, at the top
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Label("Compression Level", systemImage: "slider.horizontal.3")
                        .font(.title3.bold())
                    
                    Spacer()
                    
                    Text(compressionLabel)
                        .font(.subheadline.bold())
                        .foregroundStyle(compressionColor)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(Capsule().fill(compressionColor.opacity(0.15)))
                }
                
                VStack(spacing: 6) {
                    Slider(value: $aggressiveness, in: 0...1, step: 0.05)
                        .tint(compressionColor)
                    
                    HStack {
                        Text("0%")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(String(format: "%.0f%%", aggressiveness * 100))
                            .font(.headline.bold())
                            .foregroundStyle(compressionColor)
                        Spacer()
                        Text("100%")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(16)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(compressionColor.opacity(0.08))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12)
                                .strokeBorder(compressionColor.opacity(0.3), lineWidth: 1)
                        )
                )
            }
            
            Divider()
            
            Text("Provider")
            
            // Provider Selection
            VStack(spacing: 12) {
                ProviderOptionCard(
                    title: "Auto (Recommended)",
                    subtitle: "TokenC for short (<500 tokens), Local for long",
                    icon: "sparkles",
                    isSelected: provider == "auto",
                    color: .purple
                ) {
                    provider = "auto"
                }
                
                ProviderOptionCard(
                    title: "Local (LLMLingua)",
                    subtitle: "ML-based compression running locally",
                    icon: "desktopcomputer",
                    isSelected: provider == "local",
                    color: .green
                ) {
                    provider = "local"
                }
                
                ProviderOptionCard(
                    title: "Local (Hybrid)",
                    subtitle: "Embeddings + BM25 + heuristics",
                    icon: "cpu",
                    isSelected: provider == "hybrid",
                    color: .orange
                ) {
                    provider = "hybrid"
                }
                
                ProviderOptionCard(
                    title: "TokenC API",
                    subtitle: "Cloud-based compression service",
                    icon: "cloud.fill",
                    isSelected: provider == "tokenc",
                    color: .blue
                ) {
                    provider = "tokenc"
                }
            }
            
            Divider()
                .padding(.vertical, 4)
            
            // Provider-specific settings
            if provider == "auto" {
                VStack(alignment: .leading, spacing: 12) {
                    // TokenC settings for short prompts
                    VStack(alignment: .leading, spacing: 8) {
                        Label("TokenC API Key (for short prompts)", systemImage: "cloud.fill")
                            .font(.subheadline.bold())
                        
                        SecureField("Enter your API key", text: $tokencApiKey)
                            .textFieldStyle(.roundedBorder)
                        
                        Link("Get an API key →", destination: URL(string: "https://tokenc.com")!)
                            .font(.caption)
                            .foregroundStyle(.blue)
                    }
                    
                    // Local settings for long prompts
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Backend URL (for long prompts)", systemImage: "desktopcomputer")
                            .font(.subheadline.bold())
                        
                        TextField("http://127.0.0.1:8000/compress", text: $backendUrl)
                            .textFieldStyle(.roundedBorder)
                            .autocorrectionDisabled()
                    }
                }
            } else if provider == "tokenc" {
                VStack(alignment: .leading, spacing: 8) {
                    Text("TokenC API Key")
                        .font(.subheadline.bold())
                    
                    SecureField("Enter your API key", text: $tokencApiKey)
                        .textFieldStyle(.roundedBorder)
                    
                    Link("Get an API key →", destination: URL(string: "https://tokenc.com")!)
                        .font(.caption)
                        .foregroundStyle(.blue)
                }
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Backend URL")
                        .font(.subheadline.bold())
                    
                    TextField("http://127.0.0.1:8000/compress", text: $backendUrl)
                        .textFieldStyle(.roundedBorder)
                    .autocorrectionDisabled()
                    
                    Text("Run: cd backend && uvicorn main:app --reload")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
            
        }
        .padding(20)
        }  // ScrollView
    }
}

struct ProviderOptionCard: View {
    let title: String
    let subtitle: String
    let icon: String
    let isSelected: Bool
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .font(.system(size: 24))
                    .foregroundStyle(isSelected ? color : .secondary)
                    .frame(width: 36)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.subheadline.bold())
                        .foregroundStyle(.primary)
                    
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                
                Spacer()
                
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(size: 20))
                        .foregroundStyle(color)
                }
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(isSelected ? color.opacity(0.1) : Color.clear)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .strokeBorder(isSelected ? color : Color.secondary.opacity(0.3), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Compression Tab

struct CompressionTabView: View {
    @Binding var aggressiveness: Double
    
    private var aggressivenessLabel: String {
        switch aggressiveness {
        case 0..<0.3:
            return "Conservative"
        case 0.3..<0.6:
            return "Balanced"
        case 0.6..<0.8:
            return "Aggressive"
        default:
            return "Maximum"
        }
    }
    
    private var aggressivenessColor: Color {
        switch aggressiveness {
        case 0..<0.3:
            return .green
        case 0.3..<0.6:
            return .blue
        case 0.6..<0.8:
            return .orange
        default:
            return .red
        }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 24) {
            Text("Compression Settings")
                .font(.title3.bold())
            
            // Aggressiveness Slider
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    Text("Aggressiveness")
                        .font(.subheadline.bold())
                    
                    Spacer()
                    
                    Text(aggressivenessLabel)
                        .font(.subheadline.bold())
                        .foregroundStyle(aggressivenessColor)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(
                            Capsule()
                                .fill(aggressivenessColor.opacity(0.15))
                        )
                }
                
                VStack(spacing: 8) {
                    Slider(value: $aggressiveness, in: 0...1, step: 0.05)
                        .tint(aggressivenessColor)
                    
                    HStack {
                        Text("Keep more")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(String(format: "%.0f%%", aggressiveness * 100))
                            .font(.caption.bold())
                            .foregroundStyle(aggressivenessColor)
                        Spacer()
                        Text("Compress more")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(16)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.secondary.opacity(0.1))
                )
            }
            
            // Hot Key Info
            VStack(alignment: .leading, spacing: 8) {
                Label("Hot Key", systemImage: "keyboard")
                    .font(.subheadline.bold())
                
                HStack(spacing: 4) {
                    KeyCapView(text: "⌘")
                    Text("+")
                        .foregroundStyle(.secondary)
                    KeyCapView(text: "⌥")
                    Text("+")
                        .foregroundStyle(.secondary)
                    KeyCapView(text: "C")
                }
                
                Text("Compresses the text in the currently focused field")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.blue.opacity(0.1))
            )
            
            Spacer()
        }
        .padding(20)
    }
}

struct KeyCapView: View {
    let text: String
    
    var body: some View {
        Text(text)
            .font(.system(size: 14, weight: .medium, design: .rounded))
            .frame(minWidth: 28, minHeight: 28)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color.secondary.opacity(0.2))
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(Color.secondary.opacity(0.3), lineWidth: 1)
                    )
            )
    }
}

// MARK: - About Tab

struct AboutTabView: View {
    var body: some View {
        VStack(spacing: 20) {
            Spacer()
            
            // App Icon
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: [.cyan.opacity(0.3), .blue.opacity(0.3)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 100, height: 100)
                
                Image(systemName: "snowflake")
                    .font(.system(size: 48))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.cyan, .blue],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
            }
            
            VStack(spacing: 4) {
                Text("Polar Bear")
                    .font(.title.bold())
                
                Text("Intelligent Prompt Compression")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                
                Text("Version 1.0.0")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .padding(.top, 4)
            }
            
            Divider()
                .padding(.horizontal, 60)
            
            VStack(spacing: 12) {
                Text("Powered by LLMLingua-2")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                
                HStack(spacing: 16) {
                    Link(destination: URL(string: "https://github.com")!) {
                        Label("GitHub", systemImage: "link")
                            .font(.caption)
                    }
                    
                    Link(destination: URL(string: "https://tokenc.com")!) {
                        Label("TokenC", systemImage: "globe")
                            .font(.caption)
                    }
            }
        }
            
            Spacer()
            
            Text("© 2026 Polar Bear Team")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .padding(.bottom, 8)
        }
        .padding(20)
    }
}

// MARK: - Preview

#Preview {
    SettingsView()
}
