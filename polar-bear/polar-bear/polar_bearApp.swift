//
//  polar_bearApp.swift
//  polar-bear
//
//  Created by Yelarys Yertaiuly on 17/01/2026.
//

import SwiftUI

@main
struct polar_bearApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        Settings {
            SettingsView()
        }
    }
}
