package com.example.achats.service;

import com.example.achats.model.Categorie;
import com.example.achats.repository.CategorieRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

/**
 hedhom servicr eli yarbtou controller bel repository
 */
@Service
public class CategorieService {

    @Autowired
    private CategorieRepository categorieRepository;

    /**
     affichage des categorie lkol
     */
    public List<Categorie> findAll() {
        return categorieRepository.findAll();
    }
    /**
     affichage des categorie par id
     */
    public Optional<Categorie> findById(Long id) {
        return categorieRepository.findById(id);
    }
    /**
     sna3na categorie
     */
    public Categorie save(Categorie categorie) {
        return categorieRepository.save(categorie);
    }
    /**
     fasa5na categorie
     */
    public Categorie update(Long id, Categorie categorie) {
        Optional<Categorie> existingCategorieOpt = findById(id);
        if (existingCategorieOpt.isPresent()) {
            Categorie existingCategorie = existingCategorieOpt.get();
            // Update fields as necessary
            existingCategorie.setNom(categorie.getNom());
            // Add other fields to update as needed
            return categorieRepository.save(existingCategorie);
        }
        return null;
    }

    public void deleteById(Long id) {
        categorieRepository.deleteById(id);
    }
}
