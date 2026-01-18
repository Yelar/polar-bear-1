//
//  AppDelegate.swift
//  polar-bear
//
//  Created by Codex.
//

import AppKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let accessibilityService = AccessibilityService()
    private let hotKeyManager = GlobalHotKeyManager()
    private var statusItem: NSStatusItem?
    private let statusMenu = NSMenu()
    private var isHandlingHotKey = false

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
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = item.button {
            button.image = NSImage(systemSymbolName: "snowflake", accessibilityDescription: "Polar Bear")
        }
        item.menu = statusMenu
        statusItem = item

        statusMenu.removeAllItems()
        statusMenu.addItem(NSMenuItem(title: "Compress Focused Text", action: #selector(handleHotKeyMenu), keyEquivalent: ""))
        statusMenu.addItem(NSMenuItem.separator())
        statusMenu.addItem(NSMenuItem(title: "Settings", action: #selector(openSettings), keyEquivalent: ","))
        statusMenu.addItem(NSMenuItem(title: "Quit", action: #selector(quitApp), keyEquivalent: "q"))
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
        guard !isHandlingHotKey else { return }
        isHandlingHotKey = true

        guard let focusedElement = accessibilityService.focusedElement() else {
            isHandlingHotKey = false
            return
        }

        guard let text = accessibilityService.readText(from: focusedElement), !text.isEmpty else {
            isHandlingHotKey = false
            return
        }

        let client = NetworkClient(
            backendUrl: SettingsStore.backendUrl,
            aggressiveness: SettingsStore.aggressiveness
        )
        Task {
            do {
                let compressed = try await client.compressText(text)
                let sanitized = compressed.trimmingTrailingNewlines()
                let wrote = accessibilityService.writeText(sanitized, to: focusedElement)
                if !wrote {
                    accessibilityService.pasteTextViaKeyboard(sanitized)
                }
                if !accessibilityService.moveCaretToEnd(in: focusedElement, fallbackLength: sanitized.count) {
                    accessibilityService.moveCaretToEndViaKeyboard()
                }
            } catch {
                NSSound.beep()
            }
            isHandlingHotKey = false
        }
    }

    @objc private func openSettings() {
        NSApp.activate(ignoringOtherApps: true)
        NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
    }

    @objc private func quitApp() {
        NSApp.terminate(nil)
    }

}

enum SettingsStore {
    private static let backendUrlKey = "tokenc.backendUrl"
    private static let aggressivenessKey = "tokenc.aggressiveness"

    static var backendUrl: String {
        UserDefaults.standard.string(forKey: backendUrlKey) ?? "http://127.0.0.1:8000/compress"
    }

    static var aggressiveness: Double {
        UserDefaults.standard.double(forKey: aggressivenessKey)
    }

    static func registerDefaults() {
        UserDefaults.standard.register(defaults: [
            backendUrlKey: "http://127.0.0.1:8000/compress",
            aggressivenessKey: 0.5,
        ])
    }
}

private extension String {
    func trimmingTrailingNewlines() -> String {
        var result = self
        while let last = result.last, last == "\n" || last == "\r" {
            result.removeLast()
        }
        return result
    }
}
