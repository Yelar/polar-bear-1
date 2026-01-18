//
//  AppDelegate.swift
//  polar-bear
//
//  Created by Codex.
//

import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let accessibilityService = AccessibilityService()
    private let hotKeyManager = GlobalHotKeyManager()
    private var statusItem: NSStatusItem?
    private let statusMenu = NSMenu()
    private var isHandlingHotKey = false
    private var settingsWindow: NSWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        SettingsStore.registerDefaults()
        setupStatusItem()
        setupHotKey()
        ensureAccessibilityPermission()
    }

    func applicationWillTerminate(_ notification: Notification) {
        hotKeyManager.stop()
    }

    private func setupStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = item.button {
            // Try different SF Symbols with proper template rendering
            let symbolNames = ["snowflake", "arrow.down.left.and.arrow.up.right", "text.badge.minus", "doc.text"]
            var iconSet = false
            
            for symbolName in symbolNames {
                if let image = NSImage(systemSymbolName: symbolName, accessibilityDescription: "Polar Bear") {
                    image.isTemplate = true  // Critical: makes it adapt to dark/light menu bar
                    button.image = image
                    iconSet = true
                    print("[PolarBear] Using icon: \(symbolName)")
                    break
                }
            }
            
            if !iconSet {
                // Ultimate fallback: text title
                button.title = "🐻‍❄️"
                print("[PolarBear] Using fallback emoji icon")
            }
        }
        item.menu = statusMenu
        statusItem = item

        rebuildMenu()
    }

    private func rebuildMenu() {
        statusMenu.removeAllItems()
        
        // Token savings header
        let tokensSaved = SettingsStore.totalTokensSaved
        let headerItem = NSMenuItem(title: "💎 \(formatTokens(tokensSaved)) tokens saved", action: nil, keyEquivalent: "")
        headerItem.isEnabled = false
        statusMenu.addItem(headerItem)
        
        statusMenu.addItem(NSMenuItem.separator())
        
        let compressItem = NSMenuItem(title: "Compress Focused Text", action: #selector(handleHotKeyMenu), keyEquivalent: "")
        compressItem.keyEquivalentModifierMask = [.command, .option]
        statusMenu.addItem(compressItem)
        
        statusMenu.addItem(NSMenuItem.separator())
        
        // Provider indicator
        let providerName = SettingsStore.providerDisplayName
        let providerItem = NSMenuItem(title: "Provider: \(providerName)", action: nil, keyEquivalent: "")
        providerItem.isEnabled = false
        statusMenu.addItem(providerItem)
        
        statusMenu.addItem(NSMenuItem.separator())
        
        statusMenu.addItem(NSMenuItem(title: "Settings…", action: #selector(openSettings), keyEquivalent: ","))
        statusMenu.addItem(NSMenuItem(title: "Quit Polar Bear", action: #selector(quitApp), keyEquivalent: "q"))
    }

    private func setupHotKey() {
        hotKeyManager.onTrigger = { [weak self] in
            self?.handleHotKey()
        }
        hotKeyManager.start()
    }

    private func ensureAccessibilityPermission() {
        if !accessibilityService.ensureProcessTrusted(prompt: true) {
            accessibilityService.openAccessibilitySettings()
        }
    }

    @objc private func handleHotKeyMenu() {
        handleHotKey()
    }

    private func handleHotKey() {
        guard !isHandlingHotKey else { 
            print("[PolarBear] Already handling hotkey, ignoring")
            return 
        }
        isHandlingHotKey = true
        print("[PolarBear] Hotkey triggered!")

        guard let focusedElement = accessibilityService.focusedElement() else {
            print("[PolarBear] No focused element found")
            NSSound.beep()
            isHandlingHotKey = false
            return
        }

        guard let text = accessibilityService.readText(from: focusedElement), !text.isEmpty else {
            print("[PolarBear] No text in focused element")
            NSSound.beep()
            isHandlingHotKey = false
            return
        }
        
        print("[PolarBear] Found text (\(text.count) chars), provider: \(SettingsStore.provider)")

        let client = NetworkClient(
            backendUrl: SettingsStore.backendUrl,
            aggressiveness: SettingsStore.aggressiveness,
            provider: SettingsStore.provider,
            tokencApiKey: SettingsStore.tokencApiKey
        )
        
        Task {
            do {
                print("[PolarBear] Compressing with provider: \(client.provider)")
                let result = try await client.compressText(text)
                print("[PolarBear] Compressed: \(result.originalTokens) -> \(result.compressedTokens) tokens")
                
                let sanitized = result.output.trimmingTrailingNewlines()
                
                // Write compressed text
                let wrote = accessibilityService.writeText(sanitized, to: focusedElement)
                if !wrote {
                    print("[PolarBear] Direct write failed, using paste")
                    accessibilityService.pasteTextViaKeyboard(sanitized)
                }
                if !accessibilityService.moveCaretToEnd(in: focusedElement, fallbackLength: sanitized.count) {
                    accessibilityService.moveCaretToEndViaKeyboard()
                }
                
                // Update statistics
                SettingsStore.addTokensSaved(result.tokensSaved)
                SettingsStore.incrementCompressions()
                
                // Update menu to show new stats
                await MainActor.run {
                    rebuildMenu()
                }
                
                print("[PolarBear] Success! Saved \(result.tokensSaved) tokens")
                
            } catch {
                print("[PolarBear] Error: \(error)")
                NSSound.beep()
            }
            isHandlingHotKey = false
        }
    }

    @objc private func openSettings() {
        // If settings window exists, just show it
        if let window = settingsWindow, window.isVisible {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }
        
        // Create a new settings window with SwiftUI content
        let settingsView = SettingsView()
        let hostingController = NSHostingController(rootView: settingsView)
        
        let window = NSWindow(contentViewController: hostingController)
        window.title = "Polar Bear Settings"
        window.styleMask = [.titled, .closable, .miniaturizable]
        window.setContentSize(NSSize(width: 480, height: 520))
        window.center()
        window.isReleasedWhenClosed = false
        
        settingsWindow = window
        
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }
    
    private func formatTokens(_ count: Int) -> String {
        if count >= 1_000_000 {
            return String(format: "%.1fM", Double(count) / 1_000_000)
        } else if count >= 1_000 {
            return String(format: "%.1fK", Double(count) / 1_000)
        }
        return "\(count)"
    }
}

// MARK: - Settings Store

enum SettingsStore {
    private static let backendUrlKey = "tokenc.backendUrl"
    private static let aggressivenessKey = "tokenc.aggressiveness"
    private static let providerKey = "tokenc.provider"
    private static let tokencApiKeyKey = "tokenc.tokencApiKey"
    private static let totalTokensSavedKey = "tokenc.totalTokensSaved"
    private static let totalCompressionsKey = "tokenc.totalCompressions"

    static var backendUrl: String {
        UserDefaults.standard.string(forKey: backendUrlKey) ?? "http://127.0.0.1:8000/compress"
    }

    static var aggressiveness: Double {
        UserDefaults.standard.double(forKey: aggressivenessKey)
    }
    
    static var provider: String {
        UserDefaults.standard.string(forKey: providerKey) ?? "auto"
    }
    
    static var tokencApiKey: String {
        UserDefaults.standard.string(forKey: tokencApiKeyKey) ?? ""
    }
    
    static var totalTokensSaved: Int {
        UserDefaults.standard.integer(forKey: totalTokensSavedKey)
    }
    
    static var totalCompressions: Int {
        UserDefaults.standard.integer(forKey: totalCompressionsKey)
    }
    
    static var providerDisplayName: String {
        switch provider {
        case "auto":
            return "Auto"
        case "local":
            return "Local (LLMLingua)"
        case "hybrid":
            return "Local (Hybrid)"
        case "tokenc":
            return "TokenC API"
        default:
            return "Local"
        }
    }
    
    static func addTokensSaved(_ tokens: Int) {
        let current = UserDefaults.standard.integer(forKey: totalTokensSavedKey)
        UserDefaults.standard.set(current + tokens, forKey: totalTokensSavedKey)
    }
    
    static func incrementCompressions() {
        let current = UserDefaults.standard.integer(forKey: totalCompressionsKey)
        UserDefaults.standard.set(current + 1, forKey: totalCompressionsKey)
    }

    static func registerDefaults() {
        UserDefaults.standard.register(defaults: [
            backendUrlKey: "http://127.0.0.1:8000/compress",
            aggressivenessKey: 0.5,
            providerKey: "auto",
            tokencApiKeyKey: "",
            totalTokensSavedKey: 0,
            totalCompressionsKey: 0,
        ])
    }
}

// MARK: - String Extension

private extension String {
    func trimmingTrailingNewlines() -> String {
        var result = self
        while let last = result.last, last == "\n" || last == "\r" {
            result.removeLast()
        }
        return result
    }
}
